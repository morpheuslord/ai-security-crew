"""Threat intelligence from public sources, fetched safely and cached.

This module is the network boundary for enrichment lookups. Everything that
leaves the process goes through `fetch_bytes`, which refuses plain HTTP,
never follows redirects on its own, re-checks each hop, and stops reading a
body once it passes a byte cap. Hosts that come from user input (registry
names, token realms) are checked with `check_public_host` so a request can
never be steered at loopback, private, link-local, or metadata addresses.

Responses are cached as plain JSON files, and any text taken from a feed is
passed through `untrusted` before it is returned to an agent.
"""

import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import httpx

from mcp_security_review.guard.clip import clip

logger = logging.getLogger(__name__)

# ---- HTTP ----

MAX_REDIRECTS = 3
MAX_FETCH_SECONDS = 60.0
"""Wall-clock cap for one fetch, redirects included, so a slow drip ends."""

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DROPPED_BODY_HEADERS = ("content-encoding", "content-length", "transfer-encoding")


class FetchError(Exception):
    """A fetch failed; `code` says why in a short machine-readable form.

    Codes: `timeout`, `connect_failed`, `http_<status>`, `rate_limited`,
    `response_too_large`, `invalid_json`, `unsafe_host`, `too_many_redirects`.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Store the failure code and an optional human-readable message."""
        super().__init__(message or code)
        self.code = code


def new_client(timeout: float = 15.0) -> httpx.AsyncClient:
    """Build an HTTP client with safe defaults for intel lookups.

    Args:
        timeout: Read, write, and pool timeout in seconds. Connect is 5s.

    Returns:
        An `httpx.AsyncClient` with TLS verification on and redirects off.
    """
    # Imported here: the package __init__ pulls in the CLI and server modules.
    from mcp_security_review import __version__

    return httpx.AsyncClient(
        verify=True,
        follow_redirects=False,
        timeout=httpx.Timeout(timeout, connect=5.0),
        headers={
            "User-Agent": f"mcp-security-review/{__version__}",
            # No compression: a small gzip body can inflate far past max_bytes
            # before the cap sees it, so ask for the raw size up front.
            "Accept-Encoding": "identity",
        },
    )


async def _check_url(url: httpx.URL, *, guard_host: bool) -> None:
    """Refuse non-HTTPS URLs and, when guarding, non-public hosts."""
    if url.scheme != "https":
        raise FetchError("unsafe_host", f"only https is allowed, got {url.scheme!r}")
    if not guard_host:
        return
    if url.userinfo:
        raise FetchError("unsafe_host", "credentials in URL are not allowed")
    if url.port not in (None, 443):
        raise FetchError("unsafe_host", f"port {url.port} is not allowed")
    await _guard_host(url.host)


async def _guard_host(host: str) -> None:
    """Run `check_public_host`, reporting a refusal as `unsafe_host`."""
    try:
        await check_public_host(host)
    except UnsafeHostError as exc:
        raise FetchError("unsafe_host", str(exc)) from exc


async def _read_capped(resp: httpx.Response, max_bytes: int) -> bytes:
    """Read a streamed body, stopping as soon as it exceeds `max_bytes`."""
    declared = resp.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise FetchError("response_too_large", f"declared {declared} bytes")
    body = bytearray()
    async for chunk in resp.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise FetchError("response_too_large", f"over {max_bytes} bytes")
    return bytes(body)


async def _fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
    headers: dict[str, str] | None,
    method: str,
    guard_host: bool,
) -> httpx.Response:
    """Do the request loop for `fetch_bytes`, without the overall deadline."""
    current = httpx.URL(url)
    send_headers = dict(headers or {})
    for hop in range(MAX_REDIRECTS + 1):
        await _check_url(current, guard_host=guard_host)
        request = client.build_request(method, current, headers=send_headers)
        resp = await client.send(request, stream=True)
        try:
            if resp.status_code in _REDIRECT_STATUSES:
                location = resp.headers.get("location")
                if not location:
                    code = f"http_{resp.status_code}"
                    raise FetchError(code, "redirect without location")
                if hop == MAX_REDIRECTS:
                    raise FetchError("too_many_redirects", f"more than {MAX_REDIRECTS}")
                target = current.join(location)
                if target.host != current.host and not guard_host:
                    # Even fixed feeds must not bounce us onto an internal host.
                    await _guard_host(target.host)
                if (target.host, target.port) != (current.host, current.port):
                    # Never hand credentials to an origin the caller did not name.
                    send_headers = {
                        k: v
                        for k, v in send_headers.items()
                        if k.lower() not in ("authorization", "cookie")
                    }
                if resp.status_code == 303:
                    method = "GET"
                current = target
                continue
            if resp.status_code == 429:
                raise FetchError("rate_limited", "upstream returned 429")
            if resp.status_code >= 400:
                code = f"http_{resp.status_code}"
                raise FetchError(code)
            body = await _read_capped(resp, max_bytes)
        finally:
            await resp.aclose()
        kept = [
            (k, v)
            for k, v in resp.headers.multi_items()
            if k.lower() not in _DROPPED_BODY_HEADERS
        ]
        return httpx.Response(
            resp.status_code, headers=kept, content=body, request=request
        )
    raise FetchError("too_many_redirects", f"more than {MAX_REDIRECTS}")


async def fetch_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    guard_host: bool = False,
) -> httpx.Response:
    """Fetch an HTTPS URL with a byte cap and manual, re-checked redirects.

    Args:
        client: Client from `new_client` (or a test client).
        url: HTTPS URL to fetch.
        max_bytes: Largest decoded body accepted.
        headers: Extra request headers. Authorization and Cookie are dropped
            when a redirect moves to a different host.
        method: HTTP method.
        guard_host: Check every hop with `check_public_host` and refuse
            credentials in the URL or a port other than 443. Use it whenever
            the host comes from user or upstream input. Without it, a
            redirect to a different host is still checked.

    Returns:
        A fully read response whose `content` is the decoded body.

    Raises:
        FetchError: On any failure; see the class for codes.
    """
    try:
        with anyio.fail_after(MAX_FETCH_SECONDS):
            return await _fetch(
                client,
                url,
                max_bytes=max_bytes,
                headers=headers,
                method=method,
                guard_host=guard_host,
            )
    except TimeoutError as exc:
        raise FetchError(
            "timeout", f"no full response in {MAX_FETCH_SECONDS}s"
        ) from exc
    except httpx.TimeoutException as exc:
        raise FetchError("timeout", type(exc).__name__) from exc
    except httpx.TransportError as exc:
        raise FetchError("connect_failed", type(exc).__name__) from exc
    except httpx.InvalidURL as exc:
        raise FetchError("unsafe_host", "invalid URL") from exc
    except httpx.HTTPError as exc:  # e.g. DecodingError on a corrupt gzip body
        raise FetchError("connect_failed", type(exc).__name__) from exc


async def fetch_json(
    client: httpx.AsyncClient, url: str, *, max_bytes: int, **kw: Any
) -> Any:
    """Fetch a URL with `fetch_bytes` and parse the body as JSON.

    Args:
        client: Client from `new_client`.
        url: HTTPS URL to fetch.
        max_bytes: Largest body accepted.
        **kw: Passed through to `fetch_bytes`.

    Returns:
        The parsed JSON value.

    Raises:
        FetchError: As `fetch_bytes`, or `invalid_json` if parsing fails.
    """
    resp = await fetch_bytes(client, url, max_bytes=max_bytes, **kw)
    try:
        return json.loads(resp.content)
    # ValueError covers JSONDecodeError and UnicodeDecodeError; deeply nested
    # input makes the parser raise RecursionError instead.
    except (ValueError, RecursionError) as exc:
        raise FetchError("invalid_json", "response is not valid JSON") from exc


# ---- Host guard ----

HOSTNAME_RE = r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
_HOSTNAME = re.compile(HOSTNAME_RE)
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")
# Names that only resolve inside a private network (cloud metadata lives here).
_PRIVATE_SUFFIXES = (
    ".internal",
    ".local",
    ".localhost",
    ".localdomain",
    ".home.arpa",
    ".arpa",
    ".lan",
)


class UnsafeHostError(ValueError):
    """A host is not a public DNS name, or resolves to a non-public address."""


def _blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Say whether an address must never be contacted."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is not None and _blocked(embedded):
            return True
    if isinstance(ip, ipaddress.IPv4Address) and ip in _SHARED_ADDRESS_SPACE:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


async def check_public_host(host: str) -> None:
    """Refuse a host unless it is a public DNS name with only public addresses.

    Args:
        host: A bare hostname: no scheme, port, user info, or brackets.

    Raises:
        UnsafeHostError: The name is malformed, is an IP literal, has a
            private-network suffix, does not resolve, or any address it
            resolves to is private, loopback, link-local, multicast,
            reserved, unspecified, or in 100.64.0.0/10.
    """
    if not host.isascii():
        msg = f"not a valid public hostname: {host[:100]!r}"
        raise UnsafeHostError(msg)
    name = host.lower()
    try:
        ipaddress.ip_address(name.strip("[]"))
    except ValueError:
        pass
    else:
        msg = f"IP literal hosts are not allowed: {host!r}"
        raise UnsafeHostError(msg)
    if not _HOSTNAME.fullmatch(name):
        msg = f"not a valid public hostname: {host[:100]!r}"
        raise UnsafeHostError(msg)
    if name.endswith(_PRIVATE_SUFFIXES):
        msg = f"private-network name: {host!r}"
        raise UnsafeHostError(msg)
    try:
        infos = await anyio.getaddrinfo(name, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        msg = f"cannot resolve {host!r}"
        raise UnsafeHostError(msg) from exc
    if not infos:
        msg = f"no addresses for {host!r}"
        raise UnsafeHostError(msg)
    for info in infos:
        addr = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as exc:
            msg = f"{host!r} resolved to {addr!r}"
            raise UnsafeHostError(msg) from exc
        if _blocked(ip):
            msg = f"{host!r} resolves to non-public address {ip}"
            raise UnsafeHostError(msg)


# ---- Cache ----

_CACHE_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")


def _default_cache_root() -> Path:
    """Pick the cache directory from the environment."""
    explicit = os.environ.get("MCP_SECURITY_REVIEW_CACHE_DIR")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "mcp-security-review"
    return Path.home() / ".cache" / "mcp-security-review"


class IntelCache:
    """A small on-disk cache of JSON values with fetch timestamps.

    Each key is stored as `<root>/<key>.json` holding
    `{"fetched_at": <ISO 8601 UTC>, "data": <value>}`. Files are written
    atomically, the directory is private (0o700), and anything unreadable,
    malformed, or dated in the future is treated as a miss.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Use `root`, or the directory named by the environment.

        Args:
            root: Cache directory. Defaults to `$MCP_SECURITY_REVIEW_CACHE_DIR`,
                then `$XDG_CACHE_HOME/mcp-security-review`, then
                `~/.cache/mcp-security-review`.
        """
        self.root: Path = root if root is not None else _default_cache_root()

    def _path(self, key: str) -> Path:
        if not _CACHE_KEY.fullmatch(key):
            msg = f"invalid cache key: {key[:100]!r}"
            raise ValueError(msg)
        return self.root / f"{key}.json"

    def _load(self, key: str) -> tuple[Any, datetime] | None:
        path = self._path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return None
        if not isinstance(entry, dict) or "data" not in entry:
            return None
        stamp = entry.get("fetched_at")
        if not isinstance(stamp, str):
            return None
        try:
            fetched_at = datetime.fromisoformat(stamp)
        except ValueError:
            return None
        if fetched_at.tzinfo is None or fetched_at > datetime.now(UTC):
            return None
        return entry["data"], fetched_at

    def get(self, key: str, ttl: timedelta) -> Any | None:
        """Return the cached value if it is younger than `ttl`, else None.

        Args:
            key: Cache key matching `^[a-z0-9][a-z0-9._-]{0,80}$`.
            ttl: Maximum age of a usable entry.

        Raises:
            ValueError: The key is malformed.
        """
        loaded = self._load(key)
        if loaded is None:
            return None
        data, fetched_at = loaded
        if datetime.now(UTC) - fetched_at >= ttl:
            return None
        return data

    def get_stale(self, key: str) -> tuple[Any, datetime] | None:
        """Return the cached value and its fetch time regardless of age.

        Use this as a fallback when a refresh fails.

        Raises:
            ValueError: The key is malformed.
        """
        return self._load(key)

    def put(self, key: str, value: Any) -> None:
        """Store a JSON-serialisable value under `key`, stamped with now.

        Disk errors are logged and ignored: the cache is best effort.

        Raises:
            ValueError: The key is malformed.
            TypeError: The value cannot be serialised as JSON.
        """
        path = self._path(key)
        payload = json.dumps(
            {"fetched_at": datetime.now(UTC).isoformat(), "data": value}
        )
        tmp_name: str | None = None
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
            fd, tmp_name = tempfile.mkstemp(
                dir=self.root, prefix=f".{key}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, path)
            tmp_name = None
        except OSError as exc:
            logger.warning("Could not write intel cache %s: %s", path, exc)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass


# ---- Untrusted text ----

_UNSAFE_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"  # C0 (minus tab, LF, CR), DEL, C1
    r"\u061c\u200b-\u200f\ufeff"  # ALM, zero-width, LRM/RLM, BOM
    r"\u202a-\u202e\u2066-\u2069"  # bidi embeddings, overrides, isolates
    r"\U000e0000-\U000e007f]"  # Unicode tag characters
)


def untrusted(text: object, limit: int) -> str:
    """Make third-party text safe to hand back as data.

    Removes C0 and C1 control characters (keeping tab, newline, and carriage
    return), bidirectional controls, zero-width and byte-order marks, and
    Unicode tag characters, then clips to `limit` characters.

    Args:
        text: Any value; it is converted with `str`.
        limit: Characters to keep, not counting the omitted marker.

    Returns:
        The cleaned, clipped text.
    """
    return clip(_UNSAFE_CHARS.sub("", str(text)), limit)
