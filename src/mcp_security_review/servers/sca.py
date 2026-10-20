"""SCA (Software Composition Analysis) FastMCP server.

Provides tools for package verification and vulnerability scanning. Designed to run parallel with code
security verification during coding workflows.
"""

import json
import logging
import re
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from mcp_security_review.errors import tool_error
from mcp_security_review.providers.intel import (
    ExploitContext,
    build_exploit_contexts,
    new_client,
)
from mcp_security_review.providers.sca import OSVScanner, PackageRegistry
from mcp_security_review.providers.sca.osv import Vulnerability

logger = logging.getLogger(__name__)

_PACKAGES_EXAMPLE = '[{"name": "requests", "version": "2.31.0", "ecosystem": "pypi"}]'

_VULN_ID_RE = re.compile(
    r"^(CVE-\d{4}-\d{4,}"
    r"|GHSA(-[23456789cfghjmpqrvwx]{4}){3}"
    r"|[A-Z][A-Z0-9]*-[A-Za-z0-9._:-]{1,64})$"
)
_MAX_LOOKUP_IDS = 20


def _split_vulnerability_ids(raw: str) -> list[str]:
    """Split on commas/whitespace, drop empties, de-duplicate in order."""
    seen: set[str] = set()
    ids: list[str] = []
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part or part in seen:
            continue
        seen.add(part)
        ids.append(part)
    return ids


def _parse_packages(packages_json: str) -> list[dict] | str:
    """Parse the packages argument, or return the JSON error to send back."""
    try:
        packages = json.loads(packages_json)
    except json.JSONDecodeError as e:
        return tool_error(
            "invalid_json",
            f"packages_json is not valid JSON: {e}",
            f"Pass a JSON array, for example {_PACKAGES_EXAMPLE}",
        )
    if not isinstance(packages, list) or not all(
        isinstance(p, dict) and p.get("name") and p.get("version") for p in packages
    ):
        return tool_error(
            "invalid_packages",
            "packages_json must be a JSON array of objects with name and version.",
            f"Pass a JSON array, for example {_PACKAGES_EXAMPLE}",
        )
    return packages


sca_mcp = FastMCP(
    name="SCA MCP Service",
    description=(
        "Software Composition Analysis: package verification "
        "and vulnerability scanning."
    ),
)


@sca_mcp.tool(tags={"security", "sca", "verification"})
async def verify_packages(
    ctx: Context,
    packages_json: Annotated[
        str,
        Field(
            description=(
                "JSON array of packages to verify. Each object needs: "
                "'name' (package name), 'version' (version string), "
                "'ecosystem' ('pypi' or 'npm'). Example: "
                '[{"name": "requests", "version": "2.31.0", '
                '"ecosystem": "pypi"}]'
            )
        ),
    ],
) -> str:
    """Verify that packages exist and their versions are valid.

    Checks each package against its registry (PyPI, npm). For packages
    that don't exist or have invalid versions, suggests corrections.
    Returns only packages that need fixing — valid packages are silent.

    Use this tool when an AI coding agent introduces new dependencies
    to catch hallucinated or misspelled package names and versions.

    Args:
        ctx: The FastMCP context.
        packages_json: JSON array of package objects to verify.

    Returns:
        JSON with 'all_valid' boolean and 'invalid_packages' list.
        If all packages are valid, returns {"all_valid": true}.
        If any are invalid, returns details and suggested fixes.
    """
    packages = _parse_packages(packages_json)
    if isinstance(packages, str):
        return packages

    registry = PackageRegistry()
    results = await registry.verify_packages(packages)

    failed = [r for r in results if r.error]
    invalid = [r for r in results if not r.is_valid() and not r.error]

    if failed:
        return json.dumps(
            {
                "all_valid": False,
                "packages_checked": len(results),
                "invalid_packages": [r.to_dict() for r in invalid],
                "errors": [r.to_dict() for r in failed],
                "action_required": (
                    "Some packages could not be verified, which is not the same "
                    "as invalid. Follow each error's hint."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )

    if not invalid:
        return json.dumps(
            {
                "all_valid": True,
                "packages_checked": len(results),
            },
            indent=2,
        )

    response: dict = {
        "all_valid": False,
        "packages_checked": len(results),
        "invalid_count": len(invalid),
        "invalid_packages": [r.to_dict() for r in invalid],
        "action_required": (
            "Fix the invalid packages listed above. "
            "Use the suggested corrections where provided."
        ),
    }
    return json.dumps(response, indent=2, ensure_ascii=False)


@sca_mcp.tool(tags={"security", "sca", "vulnerability"})
async def scan_dependencies(
    ctx: Context,
    packages_json: Annotated[
        str,
        Field(
            description=(
                "JSON array of NEW packages to scan. Each object needs: "
                "'name' (package name), 'version' (version string), "
                "'ecosystem' ('pypi' or 'npm'). Example: "
                '[{"name": "pyjwt", "version": "2.4.0", '
                '"ecosystem": "pypi"}]'
            )
        ),
    ],
) -> str:
    """Scan new dependencies for known vulnerabilities using OSV.dev.

    Queries OSV.dev for CVEs affecting the given packages. Each finding
    includes a `reachability_check`: the affected symbols (when OSV has
    them) and a question. Answer it by reading your own code. If the
    code does not use the affected symbols, the vulnerability is likely
    not reached; if it does, or you cannot tell, upgrade to the fixed
    version.

    Run this whenever new packages are added. No code is sent to this
    tool.

    Args:
        ctx: The FastMCP context.
        packages_json: JSON array of package objects to scan.

    Returns:
        JSON with scan results per package: vulnerabilities, severity,
        affected and fixed versions, and the reachability check.
    """
    packages = _parse_packages(packages_json)
    if isinstance(packages, str):
        return packages

    scanner = OSVScanner()
    results = await scanner.scan_packages(packages)

    vulnerable_results = [r for r in results if r.has_vulnerabilities]

    response: dict = {
        "packages_scanned": len(results),
        "vulnerable_count": len(vulnerable_results),
    }

    failed = [r for r in results if r.error]
    if failed:
        response["errors"] = [r.to_dict() for r in failed]

    if not vulnerable_results:
        if failed:
            response["status"] = "scan_incomplete"
            response["message"] = (
                "Some packages could not be scanned, so this is not a clean "
                "result. Retry, or tell the user the scan could not run."
            )
            return json.dumps(response, indent=2, ensure_ascii=False)
        response["status"] = "clean"
        response["message"] = "No known vulnerabilities found in scanned packages."
        return json.dumps(response, indent=2, ensure_ascii=False)

    all_vulns = [v for r in vulnerable_results for v in r.vulnerabilities]
    client = new_client()
    try:
        exploit_contexts = await build_exploit_contexts(all_vulns, client=client)
    except Exception:  # enrichment is best effort; the scan must not fail
        logger.exception("Exploit context enrichment failed")
        exploit_contexts = _unavailable_contexts(all_vulns)
    finally:
        await client.aclose()

    response["status"] = "vulnerabilities_found"
    response["results"] = [
        r.to_dict(
            {
                v.id: exploit_contexts[v.id].to_dict()
                for v in r.vulnerabilities
                if v.id in exploit_contexts
            }
        )
        for r in vulnerable_results
    ]
    response["upgrade_now"] = sorted(
        {
            v.id
            for v in all_vulns
            if (ctx := exploit_contexts.get(v.id)) is not None
            and ctx.priority == "upgrade_now"
        }
    )
    response["action_required"] = (
        "For each vulnerability, answer its reachability_check from your "
        "own code. Upgrade to the fixed version when the affected "
        "symbols are used or you cannot tell; otherwise note it and "
        "consider upgrading anyway. Upgrade every id in upgrade_now "
        "whatever the reachability answer: it is in CISA KEV or in the "
        "top 10% of EPSS."
    )
    return json.dumps(response, indent=2, ensure_ascii=False)


@sca_mcp.tool(tags={"security", "sca", "vulnerability"})
async def lookup_vulnerability(
    ctx: Context,
    vulnerability_ids: Annotated[
        str,
        Field(
            description=(
                "One or more vulnerability ids to look up directly, "
                "separated by commas or whitespace (CVE-YYYY-NNNN, "
                "GHSA-xxxx-xxxx-xxxx, or another OSV id). Up to 20 ids "
                "per call. Example: "
                '"CVE-2021-44228, GHSA-jfh8-c2jp-5v3q"'
            )
        ),
    ],
) -> str:
    """Look up known vulnerabilities by id, without scanning a package.

    Use this when a vulnerability id is already known (from a report, an
    advisory link, or a previous scan) and a package-level scan is not
    needed. Each id is checked against OSV.dev independently; a failure
    on one id does not stop the others.

    Args:
        ctx: The FastMCP context.
        vulnerability_ids: Ids to look up, comma- or whitespace-separated.

    Returns:
        JSON with `results`: one entry per requested id, in request
        order, each `{"id", "status"}` where `status` is `found`,
        `not_found`, `invalid_id`, or `lookup_failed`. A `found` entry
        also carries the vulnerability's details (minus the
        reachability_check, which needs a package context this tool
        does not have), its `affected_packages`, and `exploit_context`.
        `upgrade_now` lists ids CISA KEV or top-EPSS flags.
    """
    ids = _split_vulnerability_ids(vulnerability_ids)
    if len(ids) > _MAX_LOOKUP_IDS:
        return tool_error(
            "too_many_ids",
            f"Got {len(ids)} ids; at most {_MAX_LOOKUP_IDS} are allowed per call.",
            "Split the ids across multiple calls.",
        )

    scanner = OSVScanner()
    statuses: dict[str, str] = {}
    vulns: dict[str, Vulnerability] = {}
    for vuln_id in ids:
        if not _VULN_ID_RE.match(vuln_id):
            statuses[vuln_id] = "invalid_id"
            continue
        try:
            vuln = await scanner.lookup(vuln_id)
        except Exception:  # a bad id or upstream hiccup must not fail the batch
            logger.exception("OSV lookup failed for %s", vuln_id)
            statuses[vuln_id] = "lookup_failed"
            continue
        if vuln is None:
            statuses[vuln_id] = "not_found"
            continue
        statuses[vuln_id] = "found"
        vulns[vuln_id] = vuln

    client = new_client()
    try:
        exploit_contexts = await build_exploit_contexts(
            list(vulns.values()), client=client
        )
    except Exception:  # enrichment is best effort; the lookup must not fail
        logger.exception("Exploit context enrichment failed")
        exploit_contexts = _unavailable_contexts(list(vulns.values()))
    finally:
        await client.aclose()

    results: list[dict[str, Any]] = []
    for vuln_id in ids:
        status = statuses[vuln_id]
        if status != "found":
            results.append({"id": vuln_id, "status": status})
            continue
        vuln = vulns[vuln_id]
        context = exploit_contexts.get(vuln.id)
        entry = vuln.to_dict(context.to_dict() if context else None)
        entry.pop("reachability_check", None)
        entry["status"] = "found"
        entry["affected_packages"] = vuln.affected_packages
        results.append(entry)

    response: dict[str, Any] = {
        "success": True,
        "results": results,
        "upgrade_now": sorted(
            {
                vuln.id
                for vuln in vulns.values()
                if (context := exploit_contexts.get(vuln.id)) is not None
                and context.priority == "upgrade_now"
            }
        ),
    }
    return json.dumps(response, indent=2, ensure_ascii=False)


def _unavailable_contexts(vulns: list[Vulnerability]) -> dict[str, ExploitContext]:
    """Build an all-sources-unavailable context for each vulnerability.

    Used only when `build_exploit_contexts` itself raises, which it is
    documented not to do; this is the last-resort fallback so a bug in
    enrichment never turns a successful scan into a tool failure.
    """
    contexts: dict[str, ExploitContext] = {}
    for vuln in vulns:
        cves = vuln.cve_ids()
        contexts[vuln.id] = ExploitContext(
            cves=cves,
            kev=None,
            kev_ransomware=None,
            kev_due_date=None,
            epss=None,
            epss_percentile=None,
            priority="unknown" if not cves else "check_reachability",
            sources={"kev": "unavailable: error", "epss": "unavailable: error"},
        )
    return contexts
