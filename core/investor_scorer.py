import yaml
import random
from typing import Dict, Any, List, Tuple

# Placeholder for GraphState or relevant parts of it.
# For now, idea_details is a generic Dict.
# from configs.langgraph.pitch_loop import GraphState # Or a specific Pydantic model


def load_investor_profile(profile_path: str) -> Dict[str, Any]:
    """
    Loads an investor profile (persona and scoring_rubric) from a YAML file.

    Args:
        profile_path: Path to the investor's YAML configuration file.

    Returns:
        A dictionary containing the parsed YAML content.
        Raises FileNotFoundError if the path is invalid.
        Raises YAMLError if the YAML is malformed.
    """
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = yaml.safe_load(f)

        if (
            not isinstance(profile_data, dict)
            or "persona" not in profile_data
            or "scoring_rubric" not in profile_data
        ):
            raise ValueError(
                f"Profile at {profile_path} missing 'persona' or 'scoring_rubric'."
            )
        return profile_data
    except FileNotFoundError:
        print(f"Error: Investor profile file not found at {profile_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML from {profile_path}: {e}")
        raise
    except ValueError as e:
        print(f"Error in profile structure from {profile_path}: {e}")
        raise


def score_pitch_with_rubric(
    deck_content: str, idea_details: Dict[str, Any], rubric: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Scores a pitch based on deck content, idea details, and a scoring rubric.
    This is a simplified version with mock scoring.

    Args:
        deck_content: The Marp Markdown content string of the pitch deck.
        idea_details: Other relevant parts of the GraphState (e.g., idea_description).
                      Currently not deeply used but available for future.
        rubric: The scoring_rubric dictionary from the investor profile.

    Returns:
        A tuple containing:
            - final_score (float): The calculated weighted score (0.0 - 1.0).
            - feedback (List[str]): A list of feedback strings.
    """
    total_weighted_score = 0.0
    feedback_items: List[str] = []

    print(f"  Scoring pitch for idea: {idea_details.get('idea_name', 'N/A')}")
    print(f"  Deck content length: {len(deck_content)} characters.")

    for category, details in rubric.items():
        weight = details.get("weight", 0)
        criteria = details.get("criteria", [])  # List of criteria strings

        # Mock scoring logic for this placeholder version
        mock_category_score = random.uniform(0.5, 1.0)  # Assign a random score

        if category == "pitch_quality":
            # Simple check for pitch_quality based on deck length
            if len(deck_content) > 100:  # Arbitrary threshold for a "decent" deck
                mock_category_score = random.uniform(0.7, 1.0)
            elif len(deck_content) > 0:
                mock_category_score = random.uniform(0.4, 0.7)
            else:
                mock_category_score = random.uniform(0.0, 0.4)
            feedback_items.append(
                f"Pitch Quality (Deck length: {len(deck_content)} chars): "
                f"Assigned score {mock_category_score:.2f} (weight {weight * 100:.0f}%)."
            )
        else:
            # For other categories, use the generic mock score
            feedback_items.append(
                f"{category.replace('_', ' ').capitalize()}: "
                f"Assigned score {mock_category_score:.2f} (weight {weight * 100:.0f}%). "
                f"Criteria considered: {len(criteria)}."
            )

        total_weighted_score += mock_category_score * weight

    # Ensure score is capped at 1.0 in case of rounding or rubric issues
    final_score = min(total_weighted_score, 1.0)

    feedback_items.append(f"Overall Weighted Score: {final_score:.2f} / 1.00")

    if final_score >= 0.7:
        feedback_items.append("Assessment: Promising, recommend further consideration.")
    elif final_score >= 0.5:
        feedback_items.append(
            "Assessment: Potential but needs more work on certain areas."
        )
    else:
        feedback_items.append(
            "Assessment: Significant concerns, likely a pass for now."
        )

    return final_score, feedback_items


if __name__ == "__main__":
    # This __main__ block requires YAML files in specified paths relative
    # to CWD (e.g., project root). Assumes core.investor_scorer is importable.

    print("--- Testing Investor Scorer ---")

    # Create dummy YAML files for testing if they don't exist in a simple context
    # This is more for ad-hoc testing; proper tests use fixtures.
    import os

    if not os.path.exists("agents/investors"):
        os.makedirs("agents/investors", exist_ok=True)

    dummy_vc_yaml_path = "agents/investors/dummy_vc.yaml"
    if not os.path.exists(dummy_vc_yaml_path):
        with open(dummy_vc_yaml_path, "w", encoding="utf-8") as f:
            f.write("""
persona:
  role: "Dummy VC"
scoring_rubric:
  team: { weight: 0.3, criteria: ["Strong team"] }
  market: { weight: 0.4, criteria: ["Large market"] }
  product_solution: { weight: 0.3, criteria: ["Great product"] }
""")
        print(f"Created dummy VC profile at {dummy_vc_yaml_path}")

    try:
        print(f"\nLoading profile: {dummy_vc_yaml_path}")
        profile = load_investor_profile(dummy_vc_yaml_path)
        print(f"Profile loaded for: {profile['persona']['role']}")
        # print(f"Rubric: {profile['scoring_rubric']}")

        mock_deck = "# Title\n## Slide 1\nSome content."
        mock_idea_details = {
            "idea_name": "Test Idea for Scorer",
            "idea_description": "A great test idea.",
            "evidence_items": [{"status": "Sufficient"}],
        }

        print("\nScoring pitch with loaded rubric...")
        score, feedback = score_pitch_with_rubric(
            mock_deck, mock_idea_details, profile["scoring_rubric"]
        )

        print(f"\nFinal Score: {score:.2f}")
        print("Feedback:")
        for item in feedback:
            print(f"- {item}")

    except Exception as e:
        print(f"An error occurred in __main__: {e}")

    # Test with a non-existent file
    print("\n--- Testing non-existent profile ---")
    try:
        load_investor_profile("agents/investors/non_existent.yaml")
    except FileNotFoundError:
        print("Successfully caught FileNotFoundError for non_existent.yaml.")
    except Exception as e:
        print(f"Unexpected error for non_existent.yaml: {e}")

    # Test with malformed YAML (manual creation needed for this test)
    malformed_yaml_path = "agents/investors/malformed.yaml"
    with open(malformed_yaml_path, "w", encoding="utf-8") as f:
        f.write("persona: Role\n  scoring_rubric: - Invalid YAML")  # Indentation error
    print("\n--- Testing malformed YAML profile ---")
    try:
        load_investor_profile(malformed_yaml_path)
    except yaml.YAMLError:
        print("Successfully caught YAMLError for malformed.yaml.")
    except Exception as e:
        print(f"Unexpected error for malformed.yaml: {e}")
    finally:
        if os.path.exists(malformed_yaml_path):
            os.remove(malformed_yaml_path)
        if os.path.exists(dummy_vc_yaml_path) and "dummy" in dummy_vc_yaml_path:
            os.remove(dummy_vc_yaml_path)  # Clean up dummy

    # Test with profile missing keys
    profile_missing_keys_path = "agents/investors/missing_keys.yaml"
    with open(profile_missing_keys_path, "w", encoding="utf-8") as f:
        f.write("persona: {role: 'Test'}")  # Missing scoring_rubric
    print("\n--- Testing profile with missing keys ---")
    try:
        load_investor_profile(profile_missing_keys_path)
    except ValueError as e:
        print(f"Successfully caught ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error for missing_keys.yaml: {e}")
    finally:
        if os.path.exists(profile_missing_keys_path):
            os.remove(profile_missing_keys_path)

    print("\n--- Investor Scorer Test Finished ---")
