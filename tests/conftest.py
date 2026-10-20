"""Root pytest configuration for AI Security Crew tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_intel_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the intel cache at a temp dir and never start ATT&CK downloads.

    Review tools read the cached ATT&CK index and start a background refresh
    when it is missing. Tests must not read the developer's cache or reach
    the network, so the cache is empty unless a test fills it, and the
    refresh starter used by the tools is a no-op unless a test replaces it.
    """
    cache_dir = tmp_path_factory.mktemp("intel-cache")
    monkeypatch.setenv("MCP_SECURITY_REVIEW_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        "mcp_security_review.servers.library_support.ensure_attack_index_background",
        lambda cache=None: False,
    )
