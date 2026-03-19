"""LLM backends: MockBackend for tests, AnthropicBackend for real runs."""
from __future__ import annotations
import json
import os
from typing import Protocol


class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class MockBackend:
    """Deterministic mock backend for testing — returns structured JSON."""

    def complete(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Competitor scan response (check first — its template contains "differentiation_gap")
        if "competitive intelligence" in prompt_lower or (
            "competitive landscape" in prompt_lower and "5 dimensions" not in prompt_lower
        ):
            return json.dumps({
                "competitors": [
                    {
                        "name": "IncumbentCo",
                        "description": "Market leader with full-suite product",
                        "strengths": ["Brand recognition", "Enterprise sales"],
                        "weaknesses": ["Slow innovation", "Complex pricing"],
                        "funding": "$500M Series D"
                    },
                    {
                        "name": "StartupRival",
                        "description": "VC-backed startup targeting same segment",
                        "strengths": ["Modern UX", "Fast iteration"],
                        "weaknesses": ["Limited integrations", "Small team"],
                        "funding": "$12M Series A"
                    }
                ],
                "market_leader": "IncumbentCo",
                "differentiation_gap": "Neither player addresses the SMB segment well"
            })

        # Dimension analysis response
        if "startup analyst" in prompt_lower or "5 dimensions" in prompt_lower or (
            "analyze" in prompt_lower and "startup" in prompt_lower
        ):
            return json.dumps({
                "dimensions": [
                    {"name": "market_size", "score": 3.5, "rationale": "Addressable market ~$2B, moderate growth"},
                    {"name": "differentiation", "score": 4.0, "rationale": "Unique angle on existing problem"},
                    {"name": "technical_feasibility", "score": 4.5, "rationale": "Tech stack is mature, no novel R&D needed"},
                    {"name": "go_to_market", "score": 3.0, "rationale": "Clear initial channel but scaling unclear"},
                    {"name": "risks", "score": 2.5, "rationale": "Regulatory risk and incumbent competition"},
                ],
                "summary": "Solid idea with a defensible niche and realistic technical path."
            })

        # MVP scoping response
        if "mvp" in prompt_lower or "feature" in prompt_lower or "scope" in prompt_lower:
            return json.dumps({
                "must_have": [
                    "User authentication and onboarding",
                    "Core value-delivery workflow",
                    "Basic reporting/output"
                ],
                "nice_to_have": [
                    "Integrations with third-party tools",
                    "Advanced analytics dashboard",
                    "Team collaboration features"
                ],
                "out_of_scope": [
                    "Mobile app",
                    "API marketplace",
                    "White-label offering"
                ],
                "estimated_timeline_weeks": 8
            })

        return json.dumps({"response": "OK"})


class AnthropicBackend:
    """Real backend using the Anthropic API (requires ANTHROPIC_API_KEY)."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self._model = model
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def complete(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
