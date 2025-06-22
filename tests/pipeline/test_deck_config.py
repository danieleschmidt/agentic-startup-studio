import asyncio

from pipeline.config.settings import DeckConfig, get_settings
from pipeline.services.workflow_orchestrator import WorkflowOrchestrator


def test_deck_template_default():
    cfg = DeckConfig()
    assert cfg.template_path.endswith("templates/deck_template.marp")


def test_deck_template_env(monkeypatch, tmp_path):
    temp = tmp_path / "custom.marp"
    temp.write_text("---")
    monkeypatch.setenv("DECK_TEMPLATE_PATH", str(temp))
    cfg = DeckConfig()
    assert cfg.template_path == str(temp)


def test_orchestrator_uses_config_template(monkeypatch, tmp_path):
    temp = tmp_path / "o.marp"
    temp.write_text("---")
    monkeypatch.setenv("DECK_TEMPLATE_PATH", str(temp))
    get_settings.cache_clear()
    orch = WorkflowOrchestrator()

    captured = {}

    def fake_generate(data, path):
        captured["path"] = path
        return "content"

    monkeypatch.setattr(
        "pipeline.services.workflow_orchestrator.generate_deck_content",
        fake_generate,
    )

    state = {"idea_data": {"title": "t", "description": "d"}}
    asyncio.run(orch._process_deck_generation_stage(state))

    assert captured["path"] == str(temp)
