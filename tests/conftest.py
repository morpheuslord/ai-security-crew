"""Root pytest configuration for AI Security Crew tests."""

from pathlib import Path
from typing import Any

import pytest

from mcp_security_review.providers.attack import ATTACK_INDEX_KEY, build_index
from mcp_security_review.providers.intel import IntelCache

ATTACK_FIXTURE = Path(__file__).parent / "fixtures" / "attack-enterprise-subset.json"
"""A small ATT&CK Enterprise 20.0 STIX bundle sliced from the real one."""

# Every place a background ATT&CK refresh can be started from. Each holds its
# own reference (or looks the function up at call time), so each is replaced.
_REFRESH_STARTERS = (
    "mcp_security_review.servers.library_support.ensure_attack_index_background",
    "mcp_security_review.providers.attack.ensure_attack_index_background",
)


@pytest.fixture(autouse=True)
def _isolated_intel_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the intel cache at a temp dir and never start ATT&CK downloads.

    Review tools and dependency scans read the cached ATT&CK index and start
    a background refresh when it is missing. Tests must not read the
    developer's cache or reach the network, so the cache is empty unless a
    test fills it, and the refresh starters are no-ops unless a test asks
    for `refresh_calls`.
    """
    cache_dir = tmp_path_factory.mktemp("intel-cache")
    monkeypatch.setenv("MCP_SECURITY_REVIEW_CACHE_DIR", str(cache_dir))
    for target in _REFRESH_STARTERS:
        monkeypatch.setattr(target, lambda cache=None: False)


@pytest.fixture
def refresh_calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record the background ATT&CK refreshes code asks for (the cache passed)."""
    calls: list[object] = []

    def fake(cache: object = None) -> bool:
        calls.append(cache)
        return True

    for target in _REFRESH_STARTERS:
        monkeypatch.setattr(target, fake)
    return calls


@pytest.fixture(scope="session")
def attack_data() -> dict[str, Any]:
    """The ATT&CK index built from `ATTACK_FIXTURE`, once per session.

    Shared: tests must copy it before changing it.
    """
    return build_index(ATTACK_FIXTURE)


@pytest.fixture
def cached_attack(attack_data: dict[str, Any]) -> None:
    """Put the fixture ATT&CK index in this test's intel cache."""
    IntelCache().put(ATTACK_INDEX_KEY, attack_data)
