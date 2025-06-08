import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

from core.investor_scorer import load_investor_profile, score_pitch_with_rubric

VALID_PROFILE_CONTENT = {
    "persona": {"role": "Test Investor", "focus_areas": ["AI"]},
    "scoring_rubric": {
        "team": {"weight": 0.3, "criteria": ["Experienced team"]},
        "market": {"weight": 0.4, "criteria": ["Large market"]},
        "product_solution": {"weight": 0.3, "criteria": ["Innovative solution"]},
        # Simplify: ignore pitch_quality for some tests by setting weight to 0
        "pitch_quality": {"weight": 0.0},
    },
}

VALID_PROFILE_CONTENT_WITH_PITCH_QUALITY = {
    "persona": {"role": "Test Investor Pitch Quality", "focus_areas": ["AI"]},
    "scoring_rubric": {
        "team": {"weight": 0.25, "criteria": ["Experienced team"]},
        "market": {"weight": 0.25, "criteria": ["Large market"]},
        "product_solution": {"weight": 0.25, "criteria": ["Innovative solution"]},
        "business_model": {"weight": 0.15, "criteria": ["Clear monetization"]},
        "pitch_quality": {"weight": 0.10, "criteria": ["Clear deck"]},
    },
}


@pytest.fixture
def valid_profile_file(tmp_path: Path) -> str:
    profile_file = tmp_path / "valid_profile.yaml"
    with open(profile_file, "w", encoding="utf-8") as f:
        yaml.dump(VALID_PROFILE_CONTENT, f)
    return str(profile_file)


@pytest.fixture
def profile_with_pitch_quality_rubric(tmp_path: Path) -> str:
    profile_file = tmp_path / "profile_with_pitch_quality.yaml"
    with open(profile_file, "w", encoding="utf-8") as f:
        yaml.dump(VALID_PROFILE_CONTENT_WITH_PITCH_QUALITY, f)
    return str(profile_file)


@pytest.fixture
def malformed_profile_file(tmp_path: Path) -> str:
    profile_file = tmp_path / "malformed_profile.yaml"
    # Intentionally malformed YAML
    profile_file.write_text(
        "persona: Role\n  scoring_rubric: - Invalid YAML", encoding="utf-8"
    )
    return str(profile_file)


@pytest.fixture
def incomplete_profile_file(tmp_path: Path) -> str:
    profile_file = tmp_path / "incomplete_profile.yaml"
    with open(profile_file, "w", encoding="utf-8") as f:
        yaml.dump({"persona": {"role": "Test"}}, f)  # Missing scoring_rubric
    return str(profile_file)


class TestLoadInvestorProfile:
    def test_load_valid_profile(self, valid_profile_file: str):
        profile = load_investor_profile(valid_profile_file)
        assert profile["persona"]["role"] == "Test Investor"
        assert "team" in profile["scoring_rubric"]

    def test_load_non_existent_profile(self):
        with pytest.raises(FileNotFoundError):
            load_investor_profile("non_existent_profile.yaml")

    def test_load_malformed_profile(self, malformed_profile_file: str):
        with pytest.raises(yaml.YAMLError):
            load_investor_profile(malformed_profile_file)

    def test_load_incomplete_profile(self, incomplete_profile_file: str):
        with pytest.raises(ValueError, match="missing 'persona' or 'scoring_rubric'"):
            load_investor_profile(incomplete_profile_file)


class TestScorePitchWithRubric:
    mock_idea_details = {
        "idea_name": "SuperCool AI",
        "idea_description": "Revolutionizing widget analysis with AI.",
        "evidence_items": [{"status": "Sufficient"}],
    }

    simple_rubric = VALID_PROFILE_CONTENT["scoring_rubric"]  # pitch_quality weight is 0

    def test_score_pitch_basic_structure(self):
        deck_content = "# My Pitch\nThis is great."
        score, feedback = score_pitch_with_rubric(
            deck_content, self.mock_idea_details, self.simple_rubric
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(feedback, list)
        # Num categories + overall score line + assessment line
        assert len(feedback) == len(self.simple_rubric) + 2
        assert f"Overall Weighted Score: {score:.2f} / 1.00" in feedback[-2]

    def test_pitch_quality_scoring_long_deck(
        self, profile_with_pitch_quality_rubric: str
    ):
        profile = load_investor_profile(profile_with_pitch_quality_rubric)
        rubric = profile["scoring_rubric"]

        deck_content = "This is a long and detailed pitch deck. " * 10  # > 100 chars
        # Mock random.uniform for deterministic scores other than pitch_quality.
        # For this example, focus on pitch_quality feedback & score range.

        score, feedback = score_pitch_with_rubric(
            deck_content, self.mock_idea_details, rubric
        )

        assert 0.0 <= score <= 1.0
        # Check if pitch_quality feedback reflects a good score due to length
        pitch_quality_feedback_found = False
        for item in feedback:
            if "Pitch Quality" in item and "Deck length" in item:
                # Example: "Pitch Quality (Deck length: ...): Score ... (weight ...%)"
                # Can't assert exact score due to random, so check structure.
                assert "chars): Assigned score" in item
                pitch_quality_feedback_found = True
                break
        assert pitch_quality_feedback_found

    def test_pitch_quality_scoring_short_deck(
        self, profile_with_pitch_quality_rubric: str
    ):
        profile = load_investor_profile(profile_with_pitch_quality_rubric)
        rubric = profile["scoring_rubric"]
        deck_content = "Short."  # < 100 chars, > 0

        score, feedback = score_pitch_with_rubric(
            deck_content, self.mock_idea_details, rubric
        )
        assert 0.0 <= score <= 1.0
        pitch_quality_feedback_found = False
        for item in feedback:
            if "Pitch Quality" in item and "Deck length" in item:
                assert "chars): Assigned score" in item
                pitch_quality_feedback_found = True
                break
        assert pitch_quality_feedback_found

    def test_pitch_quality_scoring_empty_deck(
        self, profile_with_pitch_quality_rubric: str
    ):
        profile = load_investor_profile(profile_with_pitch_quality_rubric)
        rubric = profile["scoring_rubric"]
        deck_content = ""  # 0 chars

        score, feedback = score_pitch_with_rubric(
            deck_content, self.mock_idea_details, rubric
        )
        assert 0.0 <= score <= 1.0
        pitch_quality_feedback_found = False
        for item in feedback:
            if "Pitch Quality" in item and "Deck length: 0 chars" in item:
                assert "chars): Assigned score" in item
                pitch_quality_feedback_found = True
                break
        assert pitch_quality_feedback_found

    def test_feedback_assessment_levels(self):
        # This test is tricky due to random scores.
        # Ideally, mock random.uniform for deterministic assessment strings.
        # For now, acknowledge this is hard to test without deeper mocking.
        # Ensure one of the expected assessment strings is present.
        deck_content = "Some deck"
        score, feedback = score_pitch_with_rubric(
            deck_content, self.mock_idea_details, self.simple_rubric
        )

        assessment_found = False
        possible_assessments = [
            "Promising, recommend further consideration.",
            "Potential but needs more work on certain areas.",
            "Significant concerns, likely a pass for now.",
        ]
        for p_assess in possible_assessments:
            if f"Assessment: {p_assess}" in feedback[-1]:
                assessment_found = True
                break
        assert assessment_found, "No valid assessment string found in feedback."
