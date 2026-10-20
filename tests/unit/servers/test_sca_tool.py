"""Tests for the sca_scan_dependencies tool."""

import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import pytest
from fastmcp import Client

from mcp_security_review.guard.clip import OMITTED
from mcp_security_review.providers.intel import ExploitContext
from mcp_security_review.providers.sca.osv import (
    OSVScanner,
    ScanResult,
    Vulnerability,
    VulnerableFunction,
)
from mcp_security_review.servers.main import main_mcp

PKGS = json.dumps([{"name": "pyjwt", "version": "2.4.0", "ecosystem": "pypi"}])

BuildContexts = Callable[..., Awaitable[dict[str, ExploitContext]]]


async def _no_enrichment(
    vulns: list[Vulnerability], **kw: Any
) -> dict[str, ExploitContext]:
    """Default stand-in for `build_exploit_contexts`: no contexts, no network."""
    return {}


async def scan(
    results: list[ScanResult],
    *,
    build_contexts: BuildContexts | None = None,
    **args: Any,
) -> dict[str, Any]:
    async def fake(self: OSVScanner, packages: list[dict]) -> list[ScanResult]:
        return results

    with (
        patch.object(OSVScanner, "scan_packages", fake),
        patch(
            "mcp_security_review.servers.sca.build_exploit_contexts",
            build_contexts or _no_enrichment,
        ),
    ):
        async with Client(main_mcp) as client:
            result = await client.call_tool(
                "sca_scan_dependencies", {"packages_json": PKGS, **args}
            )
    items = result if isinstance(result, list) else result.content
    return json.loads(items[0].text)


@pytest.mark.anyio
async def test_vulnerability_carries_reachability_check() -> None:
    vuln = Vulnerability(
        id="GHSA-1",
        summary="s",
        severity="high",
        vulnerable_functions=[VulnerableFunction(name="decode", module="jwt")],
        fixed_versions=["2.6.0"],
    )
    data = await scan([ScanResult("pyjwt", "2.4.0", "pypi", vulnerabilities=[vuln])])
    assert data["status"] == "vulnerabilities_found"
    found = data["results"][0]["vulnerabilities"][0]
    assert found["reachability_check"]["symbols"][0]["name"] == "decode"
    assert "action_required" in data


@pytest.mark.anyio
async def test_failed_scan_is_not_reported_clean() -> None:
    data = await scan([ScanResult("pyjwt", "2.4.0", "pypi", error="OSV API error")])
    assert data["status"] == "scan_incomplete"
    assert data["errors"][0]["name"] == "pyjwt"


@pytest.mark.anyio
async def test_clean_scan() -> None:
    data = await scan([ScanResult("pyjwt", "2.6.0", "pypi")])
    assert data["status"] == "clean"


@pytest.mark.anyio
async def test_code_snippets_argument_is_gone() -> None:
    with pytest.raises(Exception):  # noqa: B017
        await scan([], code_snippets="x")


async def call_tool(tool: str, **args: Any) -> dict[str, Any]:
    async with Client(main_mcp) as client:
        result = await client.call_tool(tool, args)
    items = result if isinstance(result, list) else result.content
    return json.loads(items[0].text)


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["sca_scan_dependencies", "sca_verify_packages"])
async def test_bad_json_returns_shared_error_with_example(tool: str) -> None:
    data = await call_tool(tool, packages_json="not json")
    assert data["success"] is False
    assert data["error"]["code"] == "invalid_json"
    assert '"ecosystem": "pypi"' in data["error"]["hint"]


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["sca_scan_dependencies", "sca_verify_packages"])
async def test_wrong_shape_returns_invalid_packages(tool: str) -> None:
    data = await call_tool(tool, packages_json='["requests"]')
    assert data["error"]["code"] == "invalid_packages"


@pytest.mark.anyio
async def test_scan_failure_has_code_and_hint() -> None:
    data = await scan(
        [
            ScanResult(
                "pyjwt",
                "2.4.0",
                "pypi",
                error="OSV API error: 403",
                error_code="osv_unavailable",
            )
        ]
    )
    error = data["errors"][0]["error"]
    assert error["code"] == "osv_unavailable"
    assert error["hint"]


@pytest.mark.anyio
async def test_scan_lists_upgrade_now_and_exploit_context() -> None:
    vuln = Vulnerability(id="CVE-2021-44228", summary="s", severity="critical")
    ctx = ExploitContext(
        cves=["CVE-2021-44228"],
        kev=True,
        kev_ransomware=True,
        kev_due_date="2021-12-24",
        epss=0.97,
        epss_percentile=0.99,
        priority="upgrade_now",
        sources={},
    )

    async def fake_build(
        vulns: list[Vulnerability], **kw: Any
    ) -> dict[str, ExploitContext]:
        return {v.id: ctx for v in vulns}

    data = await scan(
        [ScanResult("pyjwt", "2.4.0", "pypi", vulnerabilities=[vuln])],
        build_contexts=fake_build,
    )
    assert data["upgrade_now"] == ["CVE-2021-44228"]
    found = data["results"][0]["vulnerabilities"][0]
    assert found["exploit_context"]["priority"] == "upgrade_now"
    assert found["exploit_context"]["kev"] is True
    assert "upgrade_now" in data["action_required"]


@pytest.mark.anyio
async def test_scan_survives_enrichment_exception() -> None:
    vuln = Vulnerability(id="CVE-2021-44228", summary="s", severity="critical")

    async def fake_build(
        vulns: list[Vulnerability], **kw: Any
    ) -> dict[str, ExploitContext]:
        raise RuntimeError("boom")

    data = await scan(
        [ScanResult("pyjwt", "2.4.0", "pypi", vulnerabilities=[vuln])],
        build_contexts=fake_build,
    )
    assert data["status"] == "vulnerabilities_found"
    found = data["results"][0]["vulnerabilities"][0]
    assert found["exploit_context"]["sources"]["kev"].startswith("unavailable")
    assert found["exploit_context"]["sources"]["epss"].startswith("unavailable")
    assert data["upgrade_now"] == []


@pytest.mark.anyio
async def test_injected_summary_is_clipped() -> None:
    vuln = Vulnerability(id="GHSA-1", summary="A" * 10_000, severity="low")

    data = await scan([ScanResult("pyjwt", "2.4.0", "pypi", vulnerabilities=[vuln])])
    summary = data["results"][0]["vulnerabilities"][0]["summary"]
    assert len(summary) <= 500 + len(OMITTED)


async def lookup(
    lookups: dict[str, Vulnerability | None | Exception],
    *,
    build_contexts: BuildContexts | None = None,
    **args: Any,
) -> dict[str, Any]:
    async def fake(self: OSVScanner, vuln_id: str) -> Vulnerability | None:
        outcome = lookups[vuln_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with (
        patch.object(OSVScanner, "lookup", fake),
        patch(
            "mcp_security_review.servers.sca.build_exploit_contexts",
            build_contexts or _no_enrichment,
        ),
    ):
        async with Client(main_mcp) as client:
            result = await client.call_tool("sca_lookup_vulnerability", args)
    items = result if isinstance(result, list) else result.content
    return json.loads(items[0].text)


@pytest.mark.anyio
async def test_lookup_found_with_exploit_context() -> None:
    vuln = Vulnerability(
        id="CVE-2021-44228",
        summary="s",
        severity="critical",
        affected_packages=[{"ecosystem": "Maven", "name": "log4j-core"}],
    )
    ctx = ExploitContext(
        cves=["CVE-2021-44228"],
        kev=True,
        kev_ransomware=True,
        kev_due_date="2021-12-24",
        epss=0.97,
        epss_percentile=0.99,
        priority="upgrade_now",
        sources={},
    )

    async def fake_build(
        vulns: list[Vulnerability], **kw: Any
    ) -> dict[str, ExploitContext]:
        return {v.id: ctx for v in vulns}

    data = await lookup(
        {"CVE-2021-44228": vuln},
        build_contexts=fake_build,
        vulnerability_ids="CVE-2021-44228",
    )
    assert data["success"] is True
    result = data["results"][0]
    assert result["id"] == "CVE-2021-44228"
    assert result["status"] == "found"
    assert "reachability_check" not in result
    assert result["affected_packages"] == [{"ecosystem": "Maven", "name": "log4j-core"}]
    assert result["exploit_context"]["priority"] == "upgrade_now"
    assert data["upgrade_now"] == ["CVE-2021-44228"]


@pytest.mark.anyio
async def test_lookup_not_found_is_not_failure() -> None:
    data = await lookup({"CVE-2099-0001": None}, vulnerability_ids="CVE-2099-0001")
    assert data["success"] is True
    assert data["results"] == [{"id": "CVE-2099-0001", "status": "not_found"}]


@pytest.mark.anyio
async def test_lookup_mixed_valid_invalid() -> None:
    data = await lookup(
        {
            "CVE-2021-44228": Vulnerability(
                id="CVE-2021-44228", summary="s", severity="low"
            )
        },
        vulnerability_ids="CVE-2021-44228, not-an-id!!",
    )
    assert data["success"] is True
    statuses = {r["id"]: r["status"] for r in data["results"]}
    assert statuses["CVE-2021-44228"] == "found"
    assert statuses["not-an-id!!"] == "invalid_id"


@pytest.mark.anyio
async def test_lookup_cap_20() -> None:
    ids = ", ".join(f"CVE-2024-{1000 + i}" for i in range(21))
    data = await lookup({}, vulnerability_ids=ids)
    assert data["success"] is False
    assert data["error"]["code"] == "too_many_ids"


@pytest.mark.anyio
async def test_lookup_osv_down_marks_lookup_failed() -> None:
    data = await lookup(
        {"CVE-2021-44228": RuntimeError("osv down")},
        vulnerability_ids="CVE-2021-44228",
    )
    assert data["success"] is True
    assert data["results"] == [{"id": "CVE-2021-44228", "status": "lookup_failed"}]


@pytest.mark.anyio
async def test_lookup_path_injection_id_rejected() -> None:
    called = False

    async def fake(self: OSVScanner, vuln_id: str) -> Vulnerability | None:
        nonlocal called
        called = True
        return None

    with patch.object(OSVScanner, "lookup", fake):
        async with Client(main_mcp) as client:
            result = await client.call_tool(
                "sca_lookup_vulnerability",
                {"vulnerability_ids": "../../v1/query"},
            )
    items = result if isinstance(result, list) else result.content
    data = json.loads(items[0].text)

    assert called is False
    assert data["results"] == [{"id": "../../v1/query", "status": "invalid_id"}]
