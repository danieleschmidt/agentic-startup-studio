"""IdeaAnalyzer: evaluates a startup idea across 5 dimensions."""
from __future__ import annotations
import json
from .models import AnalysisResult, DimensionScore
from .backends import LLMBackend, MockBackend

# Weights must sum to 1.0
DIMENSION_WEIGHTS = {
    "market_size": 0.25,
    "differentiation": 0.25,
    "technical_feasibility": 0.20,
    "go_to_market": 0.20,
    "risks": 0.10,
}

GO_NOGO_THRESHOLD = 3.0


class IdeaAnalyzer:
    """Evaluates a startup idea across 5 dimensions and produces a go/no-go decision.

    Each dimension is scored 1-5. A weighted composite score is computed.
    Default threshold for GO is 3.0.
    """

    def __init__(self, backend: LLMBackend | None = None, threshold: float = GO_NOGO_THRESHOLD):
        self.backend = backend or MockBackend()
        self.threshold = threshold

    def analyze(self, idea: str) -> AnalysisResult:
        """Run the full analysis pipeline for an idea."""
        prompt = self._build_prompt(idea)
        raw = self.backend.complete(prompt)
        return self._parse_response(idea, raw)

    def _build_prompt(self, idea: str) -> str:
        return f"""You are a sharp startup analyst. Analyze this startup idea and return ONLY valid JSON.

Idea: {idea}

Evaluate across these 5 dimensions (score each 1-5, where 5 is excellent):
- market_size: Size and growth of addressable market
- differentiation: Uniqueness vs alternatives
- technical_feasibility: How achievable with current tech
- go_to_market: Clarity and strength of distribution path
- risks: Low score = high risk (regulatory, competition, etc.)

Return JSON in this exact format:
{{
  "dimensions": [
    {{"name": "market_size", "score": <1-5>, "rationale": "<1-2 sentences>"}},
    {{"name": "differentiation", "score": <1-5>, "rationale": "<1-2 sentences>"}},
    {{"name": "technical_feasibility", "score": <1-5>, "rationale": "<1-2 sentences>"}},
    {{"name": "go_to_market", "score": <1-5>, "rationale": "<1-2 sentences>"}},
    {{"name": "risks", "score": <1-5>, "rationale": "<1-2 sentences>"}}
  ],
  "summary": "<2-3 sentence overall assessment>"
}}"""

    def _parse_response(self, idea: str, raw: str) -> AnalysisResult:
        # Strip markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        dimensions = []
        for d in data.get("dimensions", []):
            name = d["name"]
            score = float(d["score"])
            weight = DIMENSION_WEIGHTS.get(name, 0.0)
            dimensions.append(DimensionScore(
                name=name,
                score=score,
                weight=weight,
                rationale=d.get("rationale", "")
            ))

        composite = sum(d.score * d.weight for d in dimensions)
        decision = "GO" if composite >= self.threshold else "NO-GO"

        return AnalysisResult(
            idea=idea,
            dimensions=dimensions,
            composite_score=round(composite, 2),
            decision=decision,
            summary=data.get("summary", "")
        )
