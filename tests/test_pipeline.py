"""Tests for EvaluationPipeline."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from startup_studio.pipeline import EvaluationPipeline, EvaluationReport
from startup_studio.backends import MockBackend


class TestEvaluationPipeline:
    def setup_method(self):
        self.pipeline = EvaluationPipeline(backend=MockBackend())

    def test_run_returns_report(self):
        report = self.pipeline.run("SaaS analytics for e-commerce")
        assert isinstance(report, EvaluationReport)

    def test_report_has_all_components(self):
        report = self.pipeline.run("any idea")
        assert report.analysis is not None
        assert report.landscape is not None
        assert report.mvp is not None

    def test_analysis_complete(self):
        report = self.pipeline.run("any idea")
        assert report.analysis.composite_score > 0
        assert report.analysis.decision in ("GO", "NO-GO")

    def test_landscape_complete(self):
        report = self.pipeline.run("any idea")
        assert len(report.landscape.competitors) > 0

    def test_mvp_complete(self):
        report = self.pipeline.run("any idea")
        assert len(report.mvp.must_have) > 0
        assert report.mvp.estimated_timeline_weeks > 0
