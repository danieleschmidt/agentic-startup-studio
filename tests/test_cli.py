"""Tests for CLI."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from startup_studio.cli import main, render_analysis
from startup_studio.pipeline import EvaluationPipeline
from startup_studio.backends import MockBackend


class TestCLI:
    def test_main_no_args_returns_nonzero(self):
        result = main([])
        assert result == 1

    def test_analyze_command_succeeds(self, capsys):
        result = main(["analyze", "AI-powered code review for startups"])
        assert result == 0

    def test_analyze_output_contains_decision(self, capsys):
        main(["analyze", "drone delivery for rural areas"])
        captured = capsys.readouterr()
        assert "DECISION" in captured.out
        assert any(d in captured.out for d in ("GO", "NO-GO"))

    def test_analyze_output_contains_mvp(self, capsys):
        main(["analyze", "some startup idea"])
        captured = capsys.readouterr()
        assert "MVP" in captured.out
        assert "MUST HAVE" in captured.out

    def test_analyze_output_contains_competitors(self, capsys):
        main(["analyze", "some startup idea"])
        captured = capsys.readouterr()
        assert "COMPETITIVE" in captured.out

    def test_render_analysis_produces_string(self):
        pipeline = EvaluationPipeline(backend=MockBackend())
        report = pipeline.run("test idea")
        output = render_analysis(report)
        assert isinstance(output, str)
        assert len(output) > 100

    def test_threshold_flag(self, capsys):
        # With very high threshold, should always be NO-GO
        main(["analyze", "some idea", "--threshold", "5.0"])
        captured = capsys.readouterr()
        assert "NO-GO" in captured.out
