"""Full evaluation pipeline: analyze + scan competitors + scope MVP."""
from __future__ import annotations
from dataclasses import dataclass
from .backends import LLMBackend, MockBackend
from .analyzer import IdeaAnalyzer
from .competitor_scanner import CompetitorScanner
from .mvp_scoper import MVPScoper
from .models import AnalysisResult, CompetitiveLandscape, MVPScope


@dataclass
class EvaluationReport:
    """Complete evaluation report for a startup idea."""
    analysis: AnalysisResult
    landscape: CompetitiveLandscape
    mvp: MVPScope


class EvaluationPipeline:
    """Orchestrates the full startup evaluation loop."""

    def __init__(self, backend: LLMBackend | None = None):
        backend = backend or MockBackend()
        self.analyzer = IdeaAnalyzer(backend)
        self.scanner = CompetitorScanner(backend)
        self.scoper = MVPScoper(backend)

    def run(self, idea: str) -> EvaluationReport:
        """Run the full evaluation pipeline for an idea."""
        analysis = self.analyzer.analyze(idea)
        landscape = self.scanner.scan(idea)
        mvp = self.scoper.scope(idea)
        return EvaluationReport(analysis=analysis, landscape=landscape, mvp=mvp)
