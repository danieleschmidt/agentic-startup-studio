# Agentic Startup Studio

Structured startup idea evaluation pipeline. Give it an idea, get back a scored analysis, competitive landscape, and MVP scope — with a GO/NO-GO decision.

## What It Does

```
startup-studio analyze "AI-powered code review tool for junior developers"
```

Runs a 3-stage pipeline:

1. **IdeaAnalyzer** — scores the idea across 5 dimensions (1-5 each), computes a weighted composite score, and outputs a GO/NO-GO decision
2. **CompetitorScanner** — maps the competitive landscape with key players, strengths/weaknesses, and the differentiation gap
3. **MVPScoper** — breaks the idea into must-have / nice-to-have / out-of-scope features with a timeline estimate

## Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Market Size | 25% | Addressable market size and growth |
| Differentiation | 25% | Uniqueness vs alternatives |
| Technical Feasibility | 20% | Achievability with current tech |
| Go-to-Market | 20% | Clarity of distribution path |
| Risks | 10% | Low score = high risk |

**GO threshold: 3.0/5.00** (configurable)

## Quick Start

```bash
# Clone
git clone https://github.com/danieleschmidt/agentic-startup-studio
cd agentic-startup-studio

# No dependencies needed for mock mode
python3 -m startup_studio.cli analyze "your startup idea here"

# With real AI (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m startup_studio.cli analyze "your startup idea" --backend anthropic
```

Or install as a CLI tool:
```bash
pip install -e .
startup-studio analyze "your startup idea"
```

## Demo

```bash
python3 demo.py
```

Runs 3 ideas through the pipeline with MockBackend.

## Backends

| Backend | Use case | Setup |
|---------|----------|-------|
| `MockBackend` | Testing, offline dev | None — always available |
| `AnthropicBackend` | Real analysis | `pip install anthropic` + `ANTHROPIC_API_KEY` |

## Architecture

```
startup_studio/
├── models.py           # Data classes (AnalysisResult, Competitor, MVPScope, etc.)
├── backends.py         # MockBackend, AnthropicBackend
├── analyzer.py         # IdeaAnalyzer — 5-dimension scoring
├── competitor_scanner.py  # CompetitorScanner — competitive landscape
├── mvp_scoper.py       # MVPScoper — feature breakdown
├── pipeline.py         # EvaluationPipeline — orchestrates all 3
└── cli.py              # CLI: startup-studio analyze "..."
```

## Tests

```bash
python3 -m pytest tests/ -v
# 36 tests, all pass, stdlib only
```

## CLI Options

```
startup-studio analyze "idea" [options]

Options:
  --backend {mock,anthropic}  LLM backend (default: mock)
  --threshold FLOAT           GO/NO-GO score threshold (default: 3.0)
```

## License

MIT
