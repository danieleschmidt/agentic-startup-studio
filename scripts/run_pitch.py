import argparse, yaml
from configs.langgraph.pitch_loop import graph
from agents.loader import load_all_agents  # util you add later

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=20000)
ap.add_argument("--threshold", type=float, default=0.8)
args = ap.parse_args()

# load agents once per run
agents = load_all_agents()

initial_state = {"agents": agents, "token_cap": args.tokens}
final_state = graph.invoke(initial_state)
score = final_state["funding_score"]

if score >= args.threshold:
    print(f"✅ Fundable idea! score={score:.2f}")
else:
    print(f"❌ Rejected. score={score:.2f}")
