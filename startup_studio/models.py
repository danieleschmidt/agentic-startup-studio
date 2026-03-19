"""Data models for startup evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    name: str
    score: float          # 1-5
    weight: float         # 0-1, must sum to 1.0 across all dimensions
    rationale: str = ""


@dataclass
class AnalysisResult:
    """Full analysis result for a startup idea."""
    idea: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    composite_score: float = 0.0
    decision: str = ""          # "GO" or "NO-GO"
    summary: str = ""

    def go_nogo(self, threshold: float = 3.0) -> str:
        return "GO" if self.composite_score >= threshold else "NO-GO"


@dataclass
class Competitor:
    """A competitor in the landscape."""
    name: str
    description: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    funding: str = "unknown"


@dataclass
class CompetitiveLandscape:
    """Competitive landscape for a domain."""
    domain: str
    competitors: list[Competitor] = field(default_factory=list)
    market_leader: Optional[str] = None
    differentiation_gap: str = ""


@dataclass
class MVPScope:
    """MVP feature breakdown."""
    idea: str
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    estimated_timeline_weeks: int = 0
