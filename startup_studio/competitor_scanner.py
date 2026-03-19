"""CompetitorScanner: maps the competitive landscape for a domain."""
from __future__ import annotations
import json
from .models import Competitor, CompetitiveLandscape
from .backends import LLMBackend, MockBackend


class CompetitorScanner:
    """Given a domain/idea, returns a structured competitive landscape.

    Uses mock data for tests or LLM for real runs.
    """

    def __init__(self, backend: LLMBackend | None = None):
        self.backend = backend or MockBackend()

    def scan(self, domain: str) -> CompetitiveLandscape:
        """Scan competitive landscape for a domain."""
        prompt = self._build_prompt(domain)
        raw = self.backend.complete(prompt)
        return self._parse_response(domain, raw)

    def _build_prompt(self, domain: str) -> str:
        return f"""You are a competitive intelligence analyst. Map the competitive landscape for:

Domain: {domain}

Return ONLY valid JSON in this exact format:
{{
  "competitors": [
    {{
      "name": "<company name>",
      "description": "<1 sentence>",
      "strengths": ["<strength1>", "<strength2>"],
      "weaknesses": ["<weakness1>", "<weakness2>"],
      "funding": "<funding stage or amount>"
    }}
  ],
  "market_leader": "<name of dominant player>",
  "differentiation_gap": "<1-2 sentences on unmet needs or whitespace>"
}}

Include 2-4 key competitors. Be factual and concise."""

    def _parse_response(self, domain: str, raw: str) -> CompetitiveLandscape:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        competitors = []
        for c in data.get("competitors", []):
            competitors.append(Competitor(
                name=c["name"],
                description=c.get("description", ""),
                strengths=c.get("strengths", []),
                weaknesses=c.get("weaknesses", []),
                funding=c.get("funding", "unknown")
            ))

        return CompetitiveLandscape(
            domain=domain,
            competitors=competitors,
            market_leader=data.get("market_leader"),
            differentiation_gap=data.get("differentiation_gap", "")
        )
