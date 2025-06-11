import os, json, uuid
from datetime import datetime
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.loader import load_agent  # simple util you’ll add later

DB_FILE = "artifacts/pitch_loop.db"
save = SqliteSaver(DB_FILE)

# ----- node definitions ------------------------------------------------- #
def generate_ideas(state):
    ceo, cto, vprd = state["agents"]["ceo"], state["agents"]["cto"], state["agents"]["vprd"]
    ideas = ceo.run("Generate 3 startup ideas with concise 2-line summaries.")
    # simple pairwise debate
    enriched = cto.run({"ideas": ideas}) + vprd.run({"ideas": ideas})
    return {"ideas": ideas, "analyses": enriched}

def create_deck(state):
    import markdown
    idea_id = str(uuid.uuid4())
    deck_md = f"# Pitch Deck {idea_id}\n\n" + state["analyses"]
    path = f"pitches/{idea_id}/deck.marp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(deck_md)
    return {"idea_id": idea_id, "deck_path": path}

def investor_review(state):
    vc, angel = state["agents"]["vc"], state["agents"]["angel"]
    deck_path = state["deck_path"]
    deck_txt = open(deck_path).read()
    vc_score = vc.run(deck_txt)
    angel_score = angel.run(deck_txt)
    funding_score = (vc_score + angel_score) / 2
    state["funding_score"] = funding_score
    return state

# ----- graph wiring ----------------------------------------------------- #
g = StateGraph()
g.add_node("generate", generate_ideas)
g.add_node("deck", create_deck)
g.add_node("invest", investor_review)

g.add_edge("generate", "deck")
g.add_edge("deck", "invest")
g.set_entry_point("generate")
graph = g.compile(checkpointer=save)
