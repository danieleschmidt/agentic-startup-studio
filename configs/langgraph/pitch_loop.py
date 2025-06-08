from typing import List, Optional, Dict, Any

# For Python 3.8, TypedDict is in typing_extensions. For 3.9+, in typing.
# Assuming env supports typing.TypedDict or has typing_extensions installed.
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

import os  # For environment variables

# Assuming core modules are accessible via PYTHONPATH
from core.evidence_collector import EvidenceCollector
from core.deck_generator import generate_deck_content
from core.investor_scorer import load_investor_profile, score_pitch_with_rubric


# --- Configuration ---
FUND_THRESHOLD = float(os.getenv("FUND_THRESHOLD", "0.8"))


# --- Define State Schema ---
class GraphState(TypedDict):
    """
    Represents the state of the pitch loop graph.
    """

    idea_name: Optional[str]
    idea_description: Optional[str]
    current_claim: Optional[str]  # For the research phase
    evidence_items: List[Dict[str, Any]]  # List of results from EvidenceCollector
    deck_content: Optional[str]  # Content of the generated pitch deck
    investor_feedback: Optional[List[str]]
    funding_score: Optional[float]
    current_phase: str  # To track which phase (Ideate, Research, etc.)
    final_status: Optional[str]  # New field for funded/rejected status


# --- Define Nodes ---


def ideate_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node for idea generation.
    Sets a mock idea name and description.
    """
    print("--- Running Ideate Node ---")
    # In a real scenario, this would involve LLM calls or agent interactions
    # to generate or refine an idea.
    updated_state = {
        "idea_name": "AI-Powered Personalized Meal Planner",
        "idea_description": (
            "A platform that uses AI to create custom meal plans based on dietary "
            "restrictions, preferences, available ingredients, and health goals. "
            "It also generates shopping lists and provides cooking instructions."
        ),
        "current_phase": "Ideate",
        "evidence_items": [],  # Initialize for research phase
        "investor_feedback": [],  # Initialize
    }
    print(f"  Generated Idea: {updated_state['idea_name']}")
    print(f"  Description: {updated_state['idea_description'][:50]}...")
    return updated_state


def research_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node for the research phase.
    Calls the EvidenceCollector for a mock claim derived from the idea.
    """
    print("--- Running Research Node ---")
    # idea_description = state.get("idea_description", "No description provided.")
    # Derive a mock claim (this would be more sophisticated in reality)
    idea_name = state.get("idea_name", "the idea")
    mock_claim = f"The core concept of '{idea_name}' is viable and has market demand."

    print(f"  Researching claim: {mock_claim}")
    # Instantiate EvidenceCollector (using its default mock search tool for now)
    # In a real setup, the search_tool might be configured globally or passed via state
    collector = EvidenceCollector(min_citations_per_claim=2)
    evidence_result = collector.collect_and_verify_evidence(claim=mock_claim)

    print(f"  Evidence Collector Result Status: {evidence_result['status']}")
    print(f"  Found {evidence_result['search_tool_provided_count']} new sources.")

    return {
        "current_claim": mock_claim,
        "evidence_items": [evidence_result],  # Store the full result as one item
        "current_phase": "Research",
    }


def deck_generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node for pitch deck generation.
    Uses the deck_generator to create Marp content.
    """
    print("--- Running Deck Generation Node ---")
    idea_name = state.get("idea_name", "Untitled Idea")
    idea_description = state.get("idea_description", "No description provided.")
    # evidence_items = state.get("evidence_items", []) # Could be used in a real deck

    # Prepare data for the deck generator
    deck_data = {
        "title": idea_name,
        "subtitle": f"A Revolutionary Approach: {idea_description[:60]}...",
        "contact_email": "contact@futurestartup.com",
        "sections": [
            {
                "title": "The Core Problem We Solve",
                "content": (
                    f"* Current methods for ({idea_description[:30]}...) are "
                    f"inefficient.\n* Users face significant challenges."
                ),
            },
            {
                "title": f"Introducing: {idea_name}",
                "content": (
                    "* Our solution leverages cutting-edge technology.\n"
                    "* Key benefits include enhanced productivity and user "
                    "satisfaction."
                ),
            },
            {
                "title": "Our Vision & Call to Action",
                "content": (
                    "* We aim to be leaders in this new domain.\n"
                    "* Join us in shaping the future."
                ),
            },
        ],
    }

    template_path = "templates/deck_template.marp"  # Ensure this path is correct
    generated_deck_md = generate_deck_content(deck_data, template_path)

    if "Error:" in generated_deck_md:
        print(f"  Error generating deck: {generated_deck_md}")
    else:
        print(f"  Successfully generated deck for: {idea_name}")

    return {
        "deck_content": generated_deck_md,
        "current_phase": "DeckGeneration",
    }


def investor_review_node(state: GraphState) -> Dict[str, Any]:
    """
    Uses the investor_scorer to evaluate the pitch deck and idea.
    """
    print("--- Running Investor Review Node ---")
    deck_content = state.get("deck_content", "")
    idea_name = state.get("idea_name", "N/A")
    idea_description = state.get("idea_description", "")
    idea_details_for_scoring = {
        "idea_name": idea_name,
        "idea_description": idea_description,
    }

    try:
        vc_profile_path = "agents/investors/vc.yaml"
        vc_profile = load_investor_profile(vc_profile_path)
        print(f"  Using investor profile: {vc_profile['persona'].get('role', 'VC')}")
    except Exception as e:
        print(f"  Error loading VC profile: {e}. Using default feedback.")
        return {
            "investor_feedback": [f"Error loading investor profile: {e}"],
            "funding_score": 0.0,
            "current_phase": "InvestorReviewFailed",  # New phase for this error
            "final_status": "ErrorInReview",
        }

    print(f"  Scoring pitch for '{idea_name}' using VC rubric...")
    final_score, feedback_items = score_pitch_with_rubric(
        deck_content, idea_details_for_scoring, vc_profile["scoring_rubric"]
    )

    print(f"  Investor Feedback: {feedback_items}")
    print(f"  Funding Score: {final_score}")

    # TODO: Add actual Gemini Pro call here for more nuanced review.
    # qualitative_feedback = call_gemini_pro_investor_agent(...)
    # feedback_items.extend(qualitative_feedback)

    return {
        "investor_feedback": feedback_items,
        "funding_score": final_score,
        "current_phase": "InvestorReviewCompleted",  # More specific phase
    }


def funded_final_node(state: GraphState) -> Dict[str, Any]:
    """Node representing the 'funded' end state."""
    print("--- Running Funded Final Node ---")
    print("  Congratulations! The idea has met the funding threshold.")
    return {"final_status": "Funded"}


def rejected_final_node(state: GraphState) -> Dict[str, Any]:
    """Node representing the 'rejected' end state."""
    print("--- Running Rejected Final Node ---")
    print("  Unfortunately, the idea did not meet the funding threshold.")
    return {"final_status": "Rejected"}


# --- Define Conditional Edge Logic ---


def decide_funding_status(state: GraphState) -> str:
    """
    Determines the next step based on the funding score.
    """
    print("--- Running Funding Decision Logic ---")
    # Removed default to catch None explicitly for funding_score
    funding_score = state.get("funding_score")
    current_phase = state.get("current_phase")

    if current_phase == "InvestorReviewFailed" or funding_score is None:
        print("  Funding score is None or review failed, routing to rejected.")
        return "rejected"

    print(f"  Funding Score: {funding_score}, Threshold: {FUND_THRESHOLD}")
    if funding_score >= FUND_THRESHOLD:
        print("  Outcome: Funded")
        return "funded"
    else:
        print("  Outcome: Rejected")
        return "rejected"


# --- Create Graph and Compile ---

workflow = StateGraph(GraphState)

workflow.add_node("ideate", ideate_node)
workflow.add_node("research", research_node)
workflow.add_node("deck_generation", deck_generation_node)
workflow.add_node("investor_review", investor_review_node)
workflow.add_node("funded_final_node", funded_final_node)
workflow.add_node("rejected_final_node", rejected_final_node)

workflow.set_entry_point("ideate")

workflow.add_edge("ideate", "research")
workflow.add_edge("research", "deck_generation")
workflow.add_edge("deck_generation", "investor_review")

workflow.add_conditional_edges(
    "investor_review",
    decide_funding_status,
    {
        "funded": "funded_final_node",
        "rejected": "rejected_final_node",
    },
)
workflow.add_edge("funded_final_node", END)
workflow.add_edge("rejected_final_node", END)

app = workflow.compile()

# --- Basic Test Invocation ---
if __name__ == "__main__":
    print("Compiling and running the LangGraph Pitch Loop...")

    # Define an initial state (can be empty or partially filled if needed by entry node)
    initial_state: GraphState = {
        "idea_name": None,
        "idea_description": None,
        "current_claim": None,
        "evidence_items": [],
        "deck_content": None,
        "investor_feedback": [],
        "funding_score": None,
        "current_phase": "Initial",
        "final_status": None,  # Initialize new field
    }

    print(
        f"\n--- Invoking graph with default mock scoring "
        f"(FUND_THRESHOLD: {FUND_THRESHOLD}) ---"
    )
    # The mock score in investor_review_node is random(0.5, 1.0).
    # Default FUND_THRESHOLD is 0.8. So, this might go to funded or rejected.
    final_state_run1 = app.invoke(initial_state)
    print("\n--- Final State (Run 1) ---")
    for key, value in final_state_run1.items():
        if key == "evidence_items" and isinstance(value, list) and value:
            print(f"{key}: [Array of {len(value)} evidence item(s), details omitted]")
        elif isinstance(value, str) and len(value) > 100:  # Truncate long strings
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
    print(f"Final decision from state: {final_state_run1.get('final_status')}")

    # To test a specific path (e.g., rejected), manipulate FUND_THRESHOLD for this run
    # (env var is preferred for external config).
    # This part is for demo if running script directly to see branches.
    # For unit tests, mock decide_funding_status or funding_score.
    print(
        "\n--- Example: How to test different paths by changing "
        "FUND_THRESHOLD externally ---"
    )
    print(f"  To force 'rejected' more often: $ FUND_THRESHOLD=0.95 python {__file__}")
    print(f"  To force 'funded' more often:   $ FUND_THRESHOLD=0.5 python {__file__}")

    # A second run example would require more direct mocking of the score
    # for true path testing within this __main__ block.
    # The random score in investor_review_node might lead to different outcomes
    # on different runs if FUND_THRESHOLD is near the random score's mean.

    # To visualize (requires graphviz and matplotlib or other viewers)
    # try:
    #     print("\nGenerating graph diagram (pitch_loop.png)...")
    #     app.get_graph().draw_mermaid_png(output_file_path="pitch_loop.png")
    #     print("Diagram saved to pitch_loop.png (if graphviz/matplotlib installed).")
    # except Exception as e:
    #     print(f"Could not generate diagram: {e}")
    #     print("Ensure graphviz (dot) and matplotlib/pygraphviz are installed.")
