"""
CLI wrapper now imports load_all_agents correctly.
"""
import argparse, yaml, json
from agents.loader import load_all_agents
from configs.langgraph.pitch_loop import graph

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=20000)
parser.add_argument("--threshold", type=float, default=0.8)
args = parser.parse_args()

state = {"agents": load_all_agents(), "token_cap": args.tokens}
out = graph.invoke(state)

print(f"⚖️  Funding-score = {out['funding_score']:.2f}")
print("✅ Funded!" if out["funding_score"] >= args.threshold else "❌ Rejected")
