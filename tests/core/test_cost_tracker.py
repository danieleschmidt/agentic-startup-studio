import time
from unittest.mock import MagicMock

from tools import cost_tracker


def test_tokens_spent_increment():
    start = cost_tracker.TOKENS_SPENT._value.get()
    cost_tracker.TOKENS_SPENT.inc(3)
    assert cost_tracker.TOKENS_SPENT._value.get() - start == 3


def test_counter_observes_duration(monkeypatch):
    observed = []

    class Dummy:
        def observe(self, value):
            observed.append(value)

    labels_mock = MagicMock(return_value=Dummy())
    monkeypatch.setattr(cost_tracker, "_OPERATION_SECONDS", MagicMock(labels=labels_mock))

    with cost_tracker.counter("test_operation"):
        time.sleep(0.01)

    assert len(observed) == 1
    assert observed[0] > 0
