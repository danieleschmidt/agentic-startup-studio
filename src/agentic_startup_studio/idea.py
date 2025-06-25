from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class IdeaCategory(str, Enum):
    """Predefined categories for startup ideas."""

    FINTECH = "fintech"
    HEALTHTECH = "healthtech"
    EDTECH = "edtech"
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    AI_ML = "ai_ml"
    BLOCKCHAIN = "blockchain"
    CONSUMER = "consumer"
    ENTERPRISE = "enterprise"
    MARKETPLACE = "marketplace"
    UNCATEGORIZED = "uncategorized"


class Idea(BaseModel):
    """Structured representation of a startup idea."""

    title: str = Field(..., min_length=10, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)
    category: IdeaCategory = Field(default=IdeaCategory.UNCATEGORIZED)
    problem: Optional[str] = Field(default=None, max_length=1000)
    solution: Optional[str] = Field(default=None, max_length=1000)
    market: Optional[str] = Field(default=None, max_length=500)
    evidence_links: List[HttpUrl] = Field(default_factory=list)

    @validator("title", "description", "problem", "solution", "market")
    def _sanitize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if "<script" in cleaned.lower():
            raise ValueError("Potentially dangerous content detected")
        return cleaned


def validate_idea(**data: object) -> Idea:
    """Validate raw idea data and return a typed model."""

    return Idea(**data)
