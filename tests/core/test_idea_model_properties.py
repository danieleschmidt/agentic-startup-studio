import pytest
from hypothesis import given, strategies as st
from uuid import UUID

from core.models import Idea, IdeaCreate, IdeaUpdate

# Strategy for generating valid UUIDs
# Hypothesis doesn't have a built-in UUID strategy, so we generate valid strings
uuid_strategy = st.uuids()

# Strategy for generating valid arxiv links (simple regex for now)
arxiv_strategy = st.from_regex(r"https://arxiv.org/abs/[0-9]{4}\.[0-9]{5}", fullmatch=True)

# Strategy for generating lists of evidence links
evidence_strategy = st.lists(st.text(min_size=5, max_size=100).map(lambda s: f"https://example.com/{s}.pdf"), max_size=5)

# Strategy for generating valid status strings
status_strategy = st.sampled_from(["ideation", "research", "funded", "rejected"])

@given(
    id=uuid_strategy,
    arxiv=arxiv_strategy,
    evidence=evidence_strategy,
    deck_path=st.one_of(st.none(), st.text(min_size=5, max_size=100)),
    status=status_strategy
)
def test_idea_model_properties(id, arxiv, evidence, deck_path, status):
    """Test that the Idea model can be created with various valid inputs."""
    idea = Idea(
        id=id,
        arxiv=arxiv,
        evidence=evidence,
        deck_path=deck_path,
        status=status
    )
    assert isinstance(idea.id, UUID)
    assert idea.arxiv == arxiv
    assert idea.evidence == evidence
    assert idea.deck_path == deck_path
    assert idea.status == status

@given(
    arxiv=arxiv_strategy,
    evidence=evidence_strategy,
    deck_path=st.one_of(st.none(), st.text(min_size=5, max_size=100)),
    status=status_strategy
)
def test_idea_create_model_properties(arxiv, evidence, deck_path, status):
    """Test that the IdeaCreate model can be created with various valid inputs."""
    idea_create = IdeaCreate(
        arxiv=arxiv,
        evidence=evidence,
        deck_path=deck_path,
        status=status
    )
    assert idea_create.arxiv == arxiv
    assert idea_create.evidence == evidence
    assert idea_create.deck_path == deck_path
    assert idea_create.status == status

@given(
    arxiv=st.one_of(st.none(), arxiv_strategy),
    evidence=st.one_of(st.none(), evidence_strategy),
    deck_path=st.one_of(st.none(), st.text(min_size=5, max_size=100), st.just("")),
    status=st.one_of(st.none(), status_strategy)
)
def test_idea_update_model_properties(arxiv, evidence, deck_path, status):
    """Test that the IdeaUpdate model can be created with various valid inputs."""
    idea_update = IdeaUpdate(
        arxiv=arxiv,
        evidence=evidence,
        deck_path=deck_path,
        status=status
    )
    assert idea_update.arxiv == arxiv
    assert idea_update.evidence == evidence
    assert idea_update.deck_path == deck_path
    assert idea_update.status == status
