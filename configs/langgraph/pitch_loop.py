from typing import List, Optional, Dict, Any

# For Python 3.8, TypedDict is in typing_extensions. For 3.9+, in typing.
# Assuming env supports typing.TypedDict or has typing_extensions installed.
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

import os  # For environment variables
import click  # For console alert handler

# Assuming core modules are accessible via PYTHONPATH
from core.alert_manager import AlertManager  # Moved import to top
from core.evidence_collector import EvidenceCollector
from core.deck_generator import generate_deck_content
from core.investor_scorer import load_investor_profile, score_pitch_with_rubric
from core.token_budget_sentinel import TokenBudgetSentinel
from core.evidence_summarizer import summarize_evidence
from core.bias_monitor import check_text_for_bias  # Import bias monitor


# --- Configuration ---
FUND_THRESHOLD = float(os.getenv("FUND_THRESHOLD", "0.8"))
MAX_PITCH_LOOP_TOKENS = 1000  # Define max tokens for the entire loop


# Instantiate AlertManager globally
alert_manager = AlertManager(log_file_path="logs/pitch_loop_alerts.log")

# Modify TokenBudgetSentinel to use AlertManager
token_sentinel = TokenBudgetSentinel(
    max_tokens=MAX_PITCH_LOOP_TOKENS,
    alert_manager=alert_manager,
    # alert_callback=_console_alert_handler # Removed
)
# _console_alert_handler is now removed.


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
    total_tokens_consumed: int  # New field
    # token_budget_alerts: List[str]  # Removed, AlertManager is the central store
    token_budget_exceeded: bool  # New field
    evidence_summary: Optional[str]  # New field for summary
    ideation_bias_check_result: Optional[Dict[str, Any]]
    deck_bias_check_result: Optional[Dict[str, Any]]
    halted_due_to_bias: bool  # New field


# --- Define Nodes ---


def ideate_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node for idea generation.
    Sets a mock idea name and description.
    """
    print("--- Running Ideate Node ---")
    # In a real scenario, this would involve LLM calls or agent interactions
    # to generate or refine an idea.

    IDEATE_COST = 200
    current_total_tokens = state.get("total_tokens_consumed", 0)
    budget_already_exceeded = state.get("token_budget_exceeded", False)

    is_within_budget_for_step = token_sentinel.check_usage(
        current_total_tokens + IDEATE_COST, "Ideate Node"
    )
    new_total_tokens = current_total_tokens + IDEATE_COST
    # If budget was already exceeded, or this step exceeded it
    final_budget_exceeded_status = (
        budget_already_exceeded or not is_within_budget_for_step
    )

    # Node logic (mocked)
    idea_name = "AI-Powered Personalized Meal Planner"
    idea_description = (
        "An innovative application that uses AI to create personalized meal plans, "
        "generate shopping lists, and guide users through cooking processes with "
        "interactive tutorials. It aims to make healthy eating easy and accessible "
        "for everyone, especially those with busy schedules or specific dietary needs. "
        "The system learns user preferences over time."
    )
    print(f"  Generated Idea: {idea_name}")
    print(f"  Description: {idea_description[:50]}...")

    # Perform bias check on the idea description
    bias_result = check_text_for_bias(idea_description)
    critical_status = bias_result.get("is_critical", "Error")
    bias_score = bias_result.get("bias_score", -1)
    click.echo(
        f"  Ideation bias check result: Critical={critical_status}, "
        f"Score={bias_score:.2f}"
    )

    return {
        "idea_name": idea_name,
        "idea_description": idea_description,
        "current_phase": "IdeateComplete",  # Updated phase
        "evidence_items": [],
        "investor_feedback": [],
        "total_tokens_consumed": new_total_tokens,
        "token_budget_exceeded": final_budget_exceeded_status,
        "ideation_bias_check_result": bias_result,  # Store the full result
    }


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

    RESEARCH_COST = 350  # Includes EvidenceCollector's simulated cost
    current_total_tokens = state.get("total_tokens_consumed", 0)
    budget_already_exceeded = state.get("token_budget_exceeded", False)
    is_within_budget_for_step = token_sentinel.check_usage(
        current_total_tokens + RESEARCH_COST, "Research Node (incl. Evidence)"
    )
    new_total_tokens = current_total_tokens + RESEARCH_COST
    final_budget_exceeded_status = (
        budget_already_exceeded or not is_within_budget_for_step
    )

    evidence_result = collector.collect_and_verify_evidence(claim=mock_claim)
    print(f"  Evidence Collector Result Status: {evidence_result['status']}")
    print(f"  Found {evidence_result['search_tool_provided_count']} new sources.")

    return {
        "current_claim": mock_claim,
        "evidence_items": [evidence_result],
        "current_phase": "Research",
        "total_tokens_consumed": new_total_tokens,
        "token_budget_exceeded": final_budget_exceeded_status,
    }


def deck_generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node for pitch deck generation.
    Uses the deck_generator to create Marp content.
    It also now summarizes evidence before deck generation.
    """
    print("--- Running Deck Generation Node ---")
    idea_name = state.get("idea_name", "Untitled Idea")
    idea_description = state.get("idea_description", "No description provided.")
    evidence_items = state.get("evidence_items", [])

    # 1. Summarize Evidence
    print(f"  Summarizing {len(evidence_items)} evidence items...")
    # The first item in evidence_items is the EvidenceCollector's result dict.
    # This dict contains 'all_sources', which is a List[str] (URLs).
    # summarize_evidence expects List[Dict[str, Any]] with 'source_url' or 'url'.

    actual_evidence_list_for_summary = []
    if evidence_items and isinstance(evidence_items[0], dict):
        source_urls_from_collector = evidence_items[0].get("all_sources", [])
        # Wrap URLs in dicts to match summarize_evidence's expected input structure.
        actual_evidence_list_for_summary = [
            {"source_url": url} for url in source_urls_from_collector
        ]

    summary = summarize_evidence(
        actual_evidence_list_for_summary, summary_length="medium"
    )
    print(f"  Evidence Summary: {summary[:100]}...")  # Print snippet of summary

    # 2. Prepare data for the deck generator
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
                    f"satisfaction.\n\n"
                    f"**Evidence Summary:** {summary}"  # Integrate summary here
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

    DECK_GENERATION_COST = 300  # Assuming summarization cost is part of this
    current_total_tokens = state.get("total_tokens_consumed", 0)
    budget_already_exceeded = state.get("token_budget_exceeded", False)
    is_within_budget_for_step = token_sentinel.check_usage(
        current_total_tokens + DECK_GENERATION_COST, "Deck Generation Node"
    )
    new_total_tokens = current_total_tokens + DECK_GENERATION_COST
    final_budget_exceeded_status = (
        budget_already_exceeded or not is_within_budget_for_step
    )

    # 3. Perform bias check on the generated deck content
    deck_bias_result = check_text_for_bias(generated_deck_md)
    critical_status_deck = deck_bias_result.get("is_critical", "Error")
    bias_score_deck = deck_bias_result.get("bias_score", -1)
    click.echo(
        f"  Deck content bias check result: Critical={critical_status_deck}, "
        f"Score={bias_score_deck:.2f}"
    )

    return {
        "deck_content": generated_deck_md,
        "evidence_summary": summary,
        "current_phase": "DeckGenerationComplete",  # Updated phase
        "total_tokens_consumed": new_total_tokens,
        "token_budget_exceeded": final_budget_exceeded_status,
        "deck_bias_check_result": deck_bias_result,  # Store the full result
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

    INVESTOR_REVIEW_COST = 250
    current_total_tokens = state.get("total_tokens_consumed", 0)
    budget_already_exceeded = state.get("token_budget_exceeded", False)
    is_within_budget_for_step = token_sentinel.check_usage(
        current_total_tokens + INVESTOR_REVIEW_COST, "Investor Review Node"
    )
    new_total_tokens = current_total_tokens + INVESTOR_REVIEW_COST
    final_budget_exceeded_status = (
        budget_already_exceeded or not is_within_budget_for_step
    )

    return {
        "investor_feedback": feedback_items,
        "funding_score": final_score,
        "current_phase": "InvestorReviewCompleted",
        "total_tokens_consumed": new_total_tokens,
        "token_budget_exceeded": final_budget_exceeded_status,
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
# REMOVED: workflow.add_edge("deck_generation", "investor_review")
# This path is now handled by the conditional edge `decide_bias_routing` below.

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


# New node for bias revision/halt
def revise_due_to_bias_node(state: GraphState) -> Dict[str, Any]:
    click.echo("--- Running Revise Due To Bias Node ---")
    alerts = []
    if state.get("ideation_bias_check_result", {}).get("is_critical"):
        alerts.append("Critical bias detected in initial idea description.")
    if state.get("deck_bias_check_result", {}).get("is_critical"):
        alerts.append("Critical bias detected in generated deck content.")

    alert_summary = " ".join(alerts)
    click.echo(f"HALT: Content revision required. Reason(s): {alert_summary}", err=True)

    current_total_tokens = state.get("total_tokens_consumed", 0)
    REVISE_NODE_COST = 50
    new_total_tokens = current_total_tokens + REVISE_NODE_COST
    token_sentinel.check_usage(new_total_tokens, "Revise/Halt Node")

    return {
        "final_status": f"Halted: Critical Bias Detected ({alert_summary})",
        "halted_due_to_bias": True,
        "total_tokens_consumed": new_total_tokens,
        "token_budget_exceeded": new_total_tokens > token_sentinel.max_tokens,
        "current_phase": "BiasHalt",
    }


# New conditional edge function for bias routing
def decide_bias_routing(state: GraphState) -> str:
    click.echo("--- Decision: Checking for Critical Bias ---")
    ideation_check = state.get("ideation_bias_check_result", {})
    deck_check = state.get("deck_bias_check_result", {})
    ideation_critical = ideation_check.get("is_critical", False)
    deck_critical = deck_check.get("is_critical", False)

    if ideation_critical or deck_critical:
        click.echo("  Critical bias detected. Routing to revision/halt.")
        return "revise_due_to_bias"
    else:
        click.echo("  No critical bias detected. Proceeding to investor review.")
        return "proceed_to_investor_review"


# Update graph definition
workflow.add_node("revise_due_to_bias", revise_due_to_bias_node)

# Remove old direct edge from deck_generation to investor_review
# This is done by simply not adding it and adding the conditional one instead.
# If an explicit edge existed and needed removal, LangGraph's API might differ.
# Here, we are redefining the outgoing paths from deck_generation.

# The old edge was: workflow.add_edge("deck_generation", "investor_review")
# This is now replaced by the conditional edge below.

workflow.add_conditional_edges(
    "deck_generation",
    decide_bias_routing,
    {
        "revise_due_to_bias": "revise_due_to_bias",
        "proceed_to_investor_review": "investor_review",
    },
)
workflow.add_edge("revise_due_to_bias", END)

app = workflow.compile()

# --- Basic Test Invocation ---
if __name__ == "__main__":
    from unittest.mock import patch  # For mocking bias check

    # Template for initial state
    initial_state_template: GraphState = {
        "idea_name": None,
        "idea_description": None,
        "current_claim": None,
        "evidence_items": [],
        "deck_content": None,
        "investor_feedback": None,
        "funding_score": None,
        "current_phase": "Initial",
        "final_status": None,
        "total_tokens_consumed": 0,
        # "token_budget_alerts": [], # Removed
        "token_budget_exceeded": False,
        "evidence_summary": None,
        "ideation_bias_check_result": None,
        "deck_bias_check_result": None,
        "halted_due_to_bias": False,
    }

    def run_pitch_simulation(test_name: str, mock_bias_side_effect=None):
        click.echo(f"\n--- Running Pitch Loop Simulation: {test_name} ---")
        # Alerts are managed globally by alert_manager, cleared after all simulations
        current_initial_state = initial_state_template.copy()

        final_state = None
        if mock_bias_side_effect:
            # Patch 'check_text_for_bias' within the current module's namespace,
            # as this is where the nodes in this file will look it up.
            with patch(
                "configs.langgraph.pitch_loop.check_text_for_bias"
            ) as mock_check_bias:
                mock_check_bias.side_effect = mock_bias_side_effect
                side_effect_str = str(mock_bias_side_effect)
                click.echo(
                    "Mocking bias check with specific side effects for this run. "
                    f"Calls will use: {side_effect_str[:50]}..."  # Truncate
                )
                final_state = app.invoke(current_initial_state)
        else:
            click.echo("Running with actual (mocked random) bias check.")
            final_state = app.invoke(current_initial_state)

        if final_state is None:  # Should not happen if app.invoke was called
            click.echo("Error: Final state not obtained.", err=True)
            return

        # final_state["token_budget_alerts"] = token_sentinel.get_alerts() # Removed

        click.echo("\n--- Final State ---")
        for key, value in final_state.items():
            if key == "token_budget_alerts":  # Skip printing if it somehow still exists
                continue
            if (
                key in ["ideation_bias_check_result", "deck_bias_check_result"]
                and value
            ):
                click.echo(f"  {key}:")
                details = value.get("details", "N/A")
                click.echo(
                    f"    Score: {value.get('bias_score', 'N/A'):.4f}, "
                    f"Critical: {value.get('is_critical', 'N/A')}"
                )
                click.echo(f"    Details (truncated): {details[:100]}...")
            elif key == "evidence_items" and value:
                click.echo(f"  {key}: (Summary of {len(value)} top-level items)")
                for idx, item_group in enumerate(value):  # evidence_items is List[Dict]
                    if isinstance(item_group, dict):
                        claim = item_group.get("claim", "N/A")
                        status = item_group.get("status", "N/A")
                        num_src = len(item_group.get("all_sources", []))
                        click.echo(
                            f"    Item {idx + 1}: Claim: '{claim[:30]}...', "
                            f"Status: {status}, Sources: {num_src}"
                        )
                    else:
                        item_str = str(item_group)[:50]
                        click.echo(
                            f"    Item {idx + 1}: Unexpected format - {item_str}..."
                        )
            elif key == "deck_content" and isinstance(value, str):
                click.echo(f"  {key} (Deck Content Truncated): {value[:150]}...")
            else:
                click.echo(f"  {key}: {value}")

        click.echo("\n--- Token Budget Sentinel Report ---")
        click.echo(f"Max Pitch Loop Tokens: {MAX_PITCH_LOOP_TOKENS}")
        tokens_consumed = final_state.get("total_tokens_consumed")
        click.echo(f"Total Tokens Consumed: {tokens_consumed}")
        budget_exceeded_state = final_state.get("token_budget_exceeded")
        click.echo(f"Budget Exceeded Flag in State: {budget_exceeded_state}")
        if tokens_consumed is not None and tokens_consumed > MAX_PITCH_LOOP_TOKENS:
            click.echo(
                f"Final Tally: Total consumed ({tokens_consumed}) > budget "
                f"({MAX_PITCH_LOOP_TOKENS}). Budget definitely exceeded."
            )
        # Alerts will be printed from the global alert_manager later
        click.echo(f"--- End of Simulation: {test_name} ---")

    print("Compiling and running the LangGraph Pitch Loop...")
    alert_manager.clear_logged_alerts()  # Clear any previous run alerts

    # Test Case 1: No critical bias (relies on random or default mock behavior)
    run_pitch_simulation("Test Case 1: No Critical Bias (Random)")

    # Test Case 2: Critical bias in ideation
    ideation_critical_res = {
        "bias_score": 0.9,
        "is_critical": True,
        "details": "Mock critical bias in ideation.",
    }
    deck_ok_res = {"bias_score": 0.1, "is_critical": False, "details": "Mock deck OK."}
    run_pitch_simulation(
        "Test Case 2: Critical Bias in Ideation",
        mock_bias_side_effect=[ideation_critical_res, deck_ok_res],
    )

    # Test Case 3: Critical bias in deck generation
    ideation_ok_res = {
        "bias_score": 0.2,
        "is_critical": False,
        "details": "Mock ideation OK.",
    }
    deck_critical_res = {
        "bias_score": 0.95,
        "is_critical": True,
        "details": "Mock critical bias in deck.",
    }
    run_pitch_simulation(
        "Test Case 3: Critical Bias in Deck Generation",
        mock_bias_side_effect=[ideation_ok_res, deck_critical_res],
    )

    # Test Case 4: Both critical (ideation should halt it first)
    # deck_critical_res might not be called if ideation halts the flow
    run_pitch_simulation(
        "Test Case 4: Both Ideation and Deck Critical",
        mock_bias_side_effect=[ideation_critical_res, deck_critical_res],
    )

    # --- AdBudgetSentinel Demonstration ---
    click.echo("\n--- Demonstrating AdBudgetSentinel with Global AlertManager ---")
    from core.ad_budget_sentinel import AdBudgetSentinel  # Import here

    demo_ad_sentinel = AdBudgetSentinel(
        max_budget=100,
        campaign_id="demo_campaign_pitch_loop",
        alert_manager=alert_manager,  # Use the global alert_manager
    )
    # Simulate spend exceeding budget to trigger an alert
    demo_ad_sentinel.check_spend(current_spend=150, current_ctr=0.05)
    # Simulate spend within budget (should not trigger a new CRITICAL alert for budget)
    demo_ad_sentinel.check_spend(current_spend=50, current_ctr=0.03)
    click.echo("AdBudgetSentinel demonstration finished.")

    # --- Print All Accumulated Alerts ---
    click.echo("\n--- All Accumulated Alerts from Global Alert Manager ---")
    logged_alerts = alert_manager.get_logged_alerts()
    if logged_alerts:
        for alert_idx, alert_msg_dict in enumerate(logged_alerts):
            # Assuming alert_msg_dict is the dictionary form from AlertManager
            click.echo(
                f"- Alert {alert_idx + 1}: Level: {alert_msg_dict['level']}, "
                f"Source: {alert_msg_dict['source']}, "
                f"Message: {alert_msg_dict['message']}"
            )
    else:
        click.echo("No alerts recorded by the global alert manager.")
    alert_manager.clear_logged_alerts()  # Clean up after printing

    click.echo("\nReminder: If not using specific mock side_effects for bias checks,")
    click.echo("you might need to run multiple times to observe different paths")
    click.echo("due to the random nature of the default 'check_text_for_bias' mock.")
