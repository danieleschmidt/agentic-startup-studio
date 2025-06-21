"""
CLI wrapper now imports load_all_agents correctly.
"""
import argparse, yaml, json
from agents.loader import load_all_agents
from configs.langgraph.pitch_loop import app

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=20000)
parser.add_argument("--threshold", type=float, default=0.8)
args = parser.parse_args()

state = {"agents": load_all_agents(), "token_cap": args.tokens}
out = app.invoke(state)

funding_score = out.get("funding_score")
final_status = out.get("final_status", "Unknown")
if funding_score is not None:
    print(f"⚖️  Funding-score = {funding_score:.2f}")
else:
    print("⚖️  Funding-score not available")
print(f"Outcome: {final_status}")
