from pipeline.config.settings import InvestorConfig


def test_investor_profile_default():
    cfg = InvestorConfig()
    assert cfg.profile == "vc"
    assert cfg.get_profile_path().endswith("agents/investors/vc.yaml")


def test_investor_profile_env(monkeypatch):
    monkeypatch.setenv("INVESTOR_PROFILE", "angel")
    cfg = InvestorConfig()
    assert cfg.profile == "angel"
    assert cfg.get_profile_path().endswith("agents/investors/angel.yaml")
