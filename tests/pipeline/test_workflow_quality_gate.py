import asyncio

import pytest

from pipeline.config.settings import get_settings
from pipeline.services.workflow_orchestrator import (
    QualityGateResult,
    WorkflowOrchestrator,
)


@pytest.fixture
def orchestrator():
    return WorkflowOrchestrator()


def test_investor_quality_gate_pass(monkeypatch, orchestrator):
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "budget",
        settings.budget.model_copy(update={"funding_threshold": 0.5}),
    )
    state = {"investor_data": {"investor_score": 0.6}}
    result = asyncio.run(orchestrator._validate_investor_quality_gate(state))
    assert result == QualityGateResult.PASSED


def test_investor_quality_gate_fail(monkeypatch, orchestrator):
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "budget",
        settings.budget.model_copy(update={"funding_threshold": 0.9}),
    )
    state = {"investor_data": {"investor_score": 0.6}}
    result = asyncio.run(orchestrator._validate_investor_quality_gate(state))
    assert result == QualityGateResult.FAILED


def test_investor_profile_env(monkeypatch):
    monkeypatch.setenv("INVESTOR_PROFILE", "angel")
    get_settings.cache_clear()
    orch = WorkflowOrchestrator()
    assert orch.settings.investor.profile == "angel"
    assert orch.settings.investor.get_profile_path().endswith("angel.yaml")
