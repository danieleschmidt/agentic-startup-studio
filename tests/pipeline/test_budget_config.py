from pipeline.config.settings import BudgetConfig


def test_funding_threshold_default():
    cfg = BudgetConfig()
    assert cfg.funding_threshold == 0.8


def test_funding_threshold_env(monkeypatch):
    monkeypatch.setenv("FUND_THRESHOLD", "0.9")
    cfg = BudgetConfig()
    assert cfg.funding_threshold == 0.9
