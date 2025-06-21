import pytest
from agents.loader import load_agent


def test_load_agent_by_name():
    agent = load_agent("CEO")
    assert agent is not None
    assert agent.name == "CEO"


def test_load_agent_case_insensitive():
    agent = load_agent("vc")
    assert agent is not None
    assert agent.name == "VC"


def test_load_agent_not_found():
    missing = load_agent("nonexistent")
    assert missing is None
