"""Tests for MVPScoper."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from startup_studio.mvp_scoper import MVPScoper
from startup_studio.backends import MockBackend
from startup_studio.models import MVPScope


class TestMVPScoper:
    def setup_method(self):
        self.scoper = MVPScoper(backend=MockBackend())

    def test_scope_returns_mvp_scope(self):
        result = self.scoper.scope("AI writing assistant for marketers")
        assert isinstance(result, MVPScope)

    def test_must_have_not_empty(self):
        result = self.scoper.scope("any idea")
        assert len(result.must_have) > 0

    def test_nice_to_have_not_empty(self):
        result = self.scoper.scope("any idea")
        assert len(result.nice_to_have) > 0

    def test_out_of_scope_not_empty(self):
        result = self.scoper.scope("any idea")
        assert len(result.out_of_scope) > 0

    def test_timeline_positive(self):
        result = self.scoper.scope("any idea")
        assert result.estimated_timeline_weeks > 0

    def test_idea_preserved(self):
        idea = "On-demand dog walking platform"
        result = self.scoper.scope(idea)
        assert result.idea == idea

    def test_features_are_strings(self):
        result = self.scoper.scope("any idea")
        for f in result.must_have + result.nice_to_have + result.out_of_scope:
            assert isinstance(f, str)
