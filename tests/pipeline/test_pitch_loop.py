import importlib
from unittest import mock


def _run_pitch_loop(score: float, threshold: str) -> dict:
    import os
    os.environ["FUND_THRESHOLD"] = threshold
    module = importlib.reload(importlib.import_module("configs.langgraph.pitch_loop"))

    with mock.patch.object(module, "check_text_for_bias", return_value={"is_critical": False, "bias_score": 0.0}), \
         mock.patch.object(module, "EvidenceCollector") as mock_collector, \
         mock.patch.object(module, "summarize_evidence", return_value="summary"), \
         mock.patch.object(module, "generate_deck_content", return_value="deck"), \
         mock.patch.object(module, "score_pitch_with_rubric", return_value=(score, ["fb"])), \
         mock.patch.object(module.token_sentinel, "check_usage", return_value=True):
        mock_collector.return_value.collect_and_verify_evidence.return_value = {
            "status": "ok",
            "all_sources": ["http://src"],
            "search_tool_provided_count": 1,
        }
        state = {
            "idea_name": None,
            "idea_description": None,
            "current_claim": None,
            "evidence_items": [],
            "deck_content": None,
            "investor_feedback": None,
            "funding_score": None,
            "current_phase": "Initial",
            "final_status": None,
            "total_tokens_consumed": 0,
            "token_budget_exceeded": False,
            "evidence_summary": None,
            "ideation_bias_check_result": None,
            "deck_bias_check_result": None,
            "halted_due_to_bias": False,
        }
        return module.app.invoke(state)


def test_pitch_loop_funded(monkeypatch):
    result = _run_pitch_loop(score=0.9, threshold="0.8")
    assert result["final_status"] == "Funded"


def test_pitch_loop_rejected(monkeypatch):
    result = _run_pitch_loop(score=0.6, threshold="0.8")
    assert result["final_status"] == "Rejected"
