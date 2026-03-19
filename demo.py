#!/usr/bin/env python3
"""Demo: evaluate 3 startup ideas of varying quality."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from startup_studio.pipeline import EvaluationPipeline
from startup_studio.backends import MockBackend
from startup_studio.cli import render_analysis

IDEAS = [
    # Strong idea
    "AI-powered code review tool that explains bugs in plain English and suggests fixes — targeted at junior developers learning to code",
    # Mediocre idea  
    "A Slack bot that sends daily motivational quotes to your team",
    # Weak idea
    "Another generic social media aggregator dashboard with no clear differentiation",
]

def main():
    pipeline = EvaluationPipeline(backend=MockBackend())
    
    print("\n" + "🚀 AGENTIC STARTUP STUDIO — DEMO " + "=" * 30)
    print("Evaluating 3 startup ideas with MockBackend\n")
    
    for i, idea in enumerate(IDEAS, 1):
        print(f"\n[IDEA {i}/3]")
        report = pipeline.run(idea)
        print(render_analysis(report))
        print()

if __name__ == "__main__":
    main()
