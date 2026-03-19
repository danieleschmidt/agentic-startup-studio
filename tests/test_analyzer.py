"""Tests for IdeaAnalyzer."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from startup_studio.analyzer import IdeaAnalyzer
from startup_studio.backends import MockBackend
from startup_studio.models import AnalysisResult, DimensionScore


class TestIdeaAnalyzer:
    def setup_method(self):
        self.analyzer = IdeaAnalyzer(backend=MockBackend())

    def test_analyze_returns_analysis_result(self):
        result = self.analyzer.analyze("AI-powered code review tool for startups")
        assert isinstance(result, AnalysisResult)

    def test_analyze_has_five_dimensions(self):
        result = self.analyzer.analyze("SaaS project management for remote teams")
        assert len(result.dimensions) == 5

    def test_dimension_names_correct(self):
        result = self.analyzer.analyze("any idea")
        names = {d.name for d in result.dimensions}
        expected = {"market_size", "differentiation", "technical_feasibility", "go_to_market", "risks"}
        assert names == expected

    def test_scores_in_valid_range(self):
        result = self.analyzer.analyze("any idea")
        for d in result.dimensions:
            assert 1.0 <= d.score <= 5.0, f"{d.name} score {d.score} out of range"

    def test_weights_sum_to_one(self):
        result = self.analyzer.analyze("any idea")
        total_weight = sum(d.weight for d in result.dimensions)
        assert abs(total_weight - 1.0) < 0.001

    def test_composite_score_computed(self):
        result = self.analyzer.analyze("any idea")
        assert result.composite_score > 0

    def test_composite_score_matches_manual_calc(self):
        result = self.analyzer.analyze("any idea")
        manual = sum(d.score * d.weight for d in result.dimensions)
        assert abs(result.composite_score - manual) < 0.01

    def test_go_decision_above_threshold(self):
        result = self.analyzer.analyze("any idea")
        # MockBackend gives ~3.5 composite, default threshold 3.0
        assert result.decision in ("GO", "NO-GO")
        if result.composite_score >= 3.0:
            assert result.decision == "GO"
        else:
            assert result.decision == "NO-GO"

    def test_custom_threshold(self):
        analyzer_high = IdeaAnalyzer(backend=MockBackend(), threshold=5.0)
        result = analyzer_high.analyze("any idea")
        assert result.decision == "NO-GO"  # Nothing scores perfect 5

        analyzer_low = IdeaAnalyzer(backend=MockBackend(), threshold=1.0)
        result = analyzer_low.analyze("any idea")
        assert result.decision == "GO"

    def test_result_contains_idea(self):
        idea = "Blockchain-based pet insurance"
        result = self.analyzer.analyze(idea)
        assert result.idea == idea

    def test_summary_not_empty(self):
        result = self.analyzer.analyze("any idea")
        assert len(result.summary) > 0
