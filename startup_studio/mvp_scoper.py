"""MVPScoper: breaks a startup idea into must-have / nice-to-have / out-of-scope."""
from __future__ import annotations
import json
from .models import MVPScope
from .backends import LLMBackend, MockBackend


class MVPScoper:
    """Produces a structured MVP scope for a startup idea.

    Features are categorized into must-have, nice-to-have, and out-of-scope.
    """

    def __init__(self, backend: LLMBackend | None = None):
        self.backend = backend or MockBackend()

    def scope(self, idea: str) -> MVPScope:
        """Generate MVP scope for an idea."""
        prompt = self._build_prompt(idea)
        raw = self.backend.complete(prompt)
        return self._parse_response(idea, raw)

    def _build_prompt(self, idea: str) -> str:
        return f"""You are a senior product manager. Define the MVP scope for this startup idea.

Idea: {idea}

Think lean. The MVP should prove the core value proposition in the shortest time.

Return ONLY valid JSON in this exact format:
{{
  "must_have": ["<feature>", ...],
  "nice_to_have": ["<feature>", ...],
  "out_of_scope": ["<feature>", ...],
  "estimated_timeline_weeks": <integer>
}}

- must_have: 3-5 features essential for launch
- nice_to_have: 3-5 features for v1.1+
- out_of_scope: 3-5 features explicitly excluded from MVP
- estimated_timeline_weeks: realistic weeks for a small team (2-3 devs)"""

    def _parse_response(self, idea: str, raw: str) -> MVPScope:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return MVPScope(
            idea=idea,
            must_have=data.get("must_have", []),
            nice_to_have=data.get("nice_to_have", []),
            out_of_scope=data.get("out_of_scope", []),
            estimated_timeline_weeks=int(data.get("estimated_timeline_weeks", 0))
        )
