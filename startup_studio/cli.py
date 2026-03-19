"""CLI: startup-studio analyze "my idea" """
from __future__ import annotations
import argparse
import os
import sys


def render_analysis(report) -> str:
    """Render a full evaluation report as human-readable text."""
    a = report.analysis
    lc = report.landscape
    mvp = report.mvp

    lines = []
    lines.append("=" * 60)
    lines.append(f"  STARTUP EVALUATION: {a.idea[:50]}")
    lines.append("=" * 60)

    # --- Dimension scores ---
    lines.append("\n📊 DIMENSION SCORES")
    lines.append("-" * 40)
    for d in a.dimensions:
        bar = "█" * int(d.score) + "░" * (5 - int(d.score))
        lines.append(f"  {d.name:<22} {bar} {d.score:.1f}/5")
        if d.rationale:
            lines.append(f"  {'':22}  ↳ {d.rationale}")

    lines.append("")
    lines.append(f"  COMPOSITE SCORE: {a.composite_score:.2f}/5.00")
    decision_icon = "✅" if a.decision == "GO" else "❌"
    lines.append(f"  DECISION:        {decision_icon} {a.decision}")

    if a.summary:
        lines.append(f"\n  Summary: {a.summary}")

    # --- Competitive landscape ---
    lines.append("\n\n🏁 COMPETITIVE LANDSCAPE")
    lines.append("-" * 40)
    if lc.market_leader:
        lines.append(f"  Market leader: {lc.market_leader}")
    for c in lc.competitors:
        lines.append(f"\n  [{c.name}]  ({c.funding})")
        lines.append(f"  {c.description}")
        if c.strengths:
            lines.append(f"  ✓ " + " | ".join(c.strengths))
        if c.weaknesses:
            lines.append(f"  ✗ " + " | ".join(c.weaknesses))
    if lc.differentiation_gap:
        lines.append(f"\n  Gap: {lc.differentiation_gap}")

    # --- MVP scope ---
    lines.append("\n\n🚀 MVP SCOPE")
    lines.append("-" * 40)
    lines.append(f"  Est. timeline: {mvp.estimated_timeline_weeks} weeks")
    lines.append("\n  MUST HAVE:")
    for f in mvp.must_have:
        lines.append(f"    • {f}")
    lines.append("\n  NICE TO HAVE:")
    for f in mvp.nice_to_have:
        lines.append(f"    ○ {f}")
    lines.append("\n  OUT OF SCOPE:")
    for f in mvp.out_of_scope:
        lines.append(f"    ✗ {f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="startup-studio",
        description="Agentic startup idea evaluation pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    analyze_cmd = sub.add_parser("analyze", help="Analyze a startup idea")
    analyze_cmd.add_argument("idea", help="Startup idea description")
    analyze_cmd.add_argument(
        "--backend", choices=["mock", "anthropic"], default="mock",
        help="LLM backend to use (default: mock)"
    )
    analyze_cmd.add_argument(
        "--threshold", type=float, default=3.0,
        help="GO/NO-GO score threshold (default: 3.0)"
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "analyze":
        from .pipeline import EvaluationPipeline
        from .backends import MockBackend, AnthropicBackend
        from .analyzer import IdeaAnalyzer

        if args.backend == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
                return 1
            backend = AnthropicBackend()
        else:
            backend = MockBackend()

        pipeline = EvaluationPipeline(backend)
        # Override threshold if specified
        pipeline.analyzer.threshold = args.threshold

        print(f"Analyzing: {args.idea}")
        print("Running evaluation pipeline...\n")

        report = pipeline.run(args.idea)
        print(render_analysis(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
