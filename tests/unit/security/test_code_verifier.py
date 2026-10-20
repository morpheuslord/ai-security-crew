"""Tests for the library-driven code review context."""

import pytest

from mcp_security_review.library import UnknownComponentError
from mcp_security_review.security import CodeReviewContextBuilder


def build(**kwargs):
    return CodeReviewContextBuilder().build_review_context(**kwargs)


def test_no_components_gives_only_the_baseline_checklist() -> None:
    context = build(file_path="app/views.py")

    assert context.components == []
    assert context.review_focus_areas == []
    assert len(context.security_checklist) == 5


def test_components_target_the_checklist_at_their_threats() -> None:
    context = build(file_path="views.py", components=["file-upload"])

    assert context.components == ["file-upload"]
    assert context.risk_level == "high"
    assert any("../" in area for area in context.review_focus_areas)
    assert any(
        item.startswith("Never use the user's filename or path")
        for item in context.security_checklist
    )


def test_baseline_checks_are_always_included() -> None:
    context = build()

    joined = " ".join(context.security_checklist)
    assert "parameterized queries" in joined.lower()
    assert "secrets" in joined.lower()


def test_checklist_has_no_duplicates() -> None:
    context = build(file_path="views.py", components=["database", "file-upload"])
    assert len(context.security_checklist) == len(set(context.security_checklist))


def test_no_components_means_no_risk_level() -> None:
    context = build(file_path="views.py")
    assert context.risk_level is None
    assert "Risk Level" not in context.review_prompt


def test_sensitive_data_raises_the_risk_level() -> None:
    context = build(components=["file-upload"], data_handled=["personal_data"])
    assert context.risk_level == "critical"


def test_unknown_component_is_rejected() -> None:
    with pytest.raises(UnknownComponentError):
        build(components=["teleporter"])


def test_review_prompt_contains_the_checklist_but_no_code() -> None:
    context = build(file_path="views.py", components=["file-upload"])

    assert "```" not in context.review_prompt
    assert "already in your context" in context.review_prompt
    assert "### Security Checklist" in context.review_prompt
    assert "**Components:** file-upload" in context.review_prompt
    assert "Language" not in context.review_prompt


def test_verify_returns_iac_checklist_with_ids_and_false_positives() -> None:
    context = build(components=["cloud-compute"])

    imds = next(i for i in context.security_checklist if "AVD-AWS-0028" in i)
    assert "[Trivy AVD-AWS-0028" in imds
    assert "Checkov CKV_AWS_79" in imds
    assert "Prowler ec2_instance_imdsv2_enabled" in imds
    assert imds.endswith(
        "(false positive: Scanners may flag aws_instance when metadata_options "
        "are set on the launch template it uses.)"
    )
    joined = " ".join(context.security_checklist).lower()
    # The app-code baseline is replaced; only provider-neutral baseline items
    # about secrets and logging remain.
    assert "parameterized" not in joined
    assert "validate all input" not in joined
    assert "keep secrets and personal data out of logs" in joined
    assert "load secrets from a secrets manager" in joined


def test_iac_checklist_is_not_cut_at_the_app_limit() -> None:
    context = build(components=["container-and-iac"])

    assert context.components[0] == "cloud-network"
    assert 15 < len(context.security_checklist) <= 42
    assert any("ssm:*" in item for item in context.security_checklist)


def test_kubernetes_and_ci_use_the_iac_checklist() -> None:
    for component in ("kubernetes", "ci-pipeline", "container-image"):
        joined = " ".join(build(components=[component]).security_checklist)
        assert "parameterized" not in joined.lower(), component


def test_app_components_keep_the_app_baseline() -> None:
    joined = " ".join(build(components=["database"]).security_checklist).lower()
    assert "parameterized" in joined
    assert "[trivy" not in joined
