"""Tests for CompetitorScanner."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from startup_studio.competitor_scanner import CompetitorScanner
from startup_studio.backends import MockBackend
from startup_studio.models import CompetitiveLandscape, Competitor


class TestCompetitorScanner:
    def setup_method(self):
        self.scanner = CompetitorScanner(backend=MockBackend())

    def test_scan_returns_landscape(self):
        result = self.scanner.scan("project management software")
        assert isinstance(result, CompetitiveLandscape)

    def test_landscape_has_competitors(self):
        result = self.scanner.scan("any domain")
        assert len(result.competitors) > 0

    def test_competitor_structure(self):
        result = self.scanner.scan("any domain")
        for c in result.competitors:
            assert isinstance(c, Competitor)
            assert len(c.name) > 0
            assert isinstance(c.strengths, list)
            assert isinstance(c.weaknesses, list)

    def test_market_leader_set(self):
        result = self.scanner.scan("any domain")
        assert result.market_leader is not None
        assert len(result.market_leader) > 0

    def test_differentiation_gap_set(self):
        result = self.scanner.scan("any domain")
        assert len(result.differentiation_gap) > 0

    def test_domain_preserved(self):
        domain = "B2B HR software"
        result = self.scanner.scan(domain)
        assert result.domain == domain
