# Agentic Startup Studio

> **Vision:** Automate the entire early‑stage startup loop— ideation, validation, MVP delivery, go‑to‑market, and kill‑or‑pivot decisions—using a swarm of open‑source AI agents.

This repository seeds a production‑ready stack that strings together best‑in‑class multi‑agent frameworks (**AutoGPT** ([github.com](https://github.com/Significant-Gravitas/AutoGPT?utm_source=chatgpt.com)), **SuperAGI** ([superagi.com](https://superagi.com/open-source/?utm_source=chatgpt.com)), **MetaGPT** ([github.com](https://github.com/FoundationAgents/MetaGPT?utm_source=chatgpt.com)), **CrewAI** ([github.com](https://github.com/crewAIInc/crewAI?utm_source=chatgpt.com))) with specialised subsystems for autonomous coding (**GPT‑Engineer** ([github.com](https://github.com/AntonOsika/gpt-engineer?utm_source=chatgpt.com)), **OpenDevin** ([github.com](https://github.com/AI-App/OpenDevin.OpenDevin?utm_source=chatgpt.com))), growth automation (**Social‑GPT** ([github.com](https://github.com/Social-GPT/agent?utm_source=chatgpt.com)), **Dittofeed** ([github.com](https://github.com/dittofeed/dittofeed?utm_source=chatgpt.com))) and product analytics (**PostHog** ([github.com](https://github.com/PostHog/posthog?utm_source=chatgpt.com))). Orchestration/state is handled via **LangGraph** ([langchain.com](https://www.langchain.com/langgraph?utm_source=chatgpt.com)) and a visual no‑code canvas via **Flowise** ([github.com](https://github.com/FlowiseAI/Flowise?utm_source=chatgpt.com)).

Even conservative forecasts put the autonomous‑coding market near \$1 B in VC funding, underscoring the momentum behind agentic development ([ft.com](https://www.ft.com/content/4868bd38-613c-4fa9-ba9d-1ed8fa8a40c8?utm_source=chatgpt.com)).

---

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Repository Layout](#repository-layout)
4. [Quick Start](#quick-start)
5. [Running an Experiment](#running-an-experiment)
6. [Safety & Governance](#safety--governance)
7. [Roadmap](#roadmap)
8. [Contributing](#contributing)
9. [License](#license)

---

## Key Features

| Phase                    | Agents & Tools                                                                                                                                                                                                                                                                                     | Outcome                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Ideation**             | MetaGPT CEO/BD roles ([github.com](https://github.com/FoundationAgents/MetaGPT?utm_source=chatgpt.com))                                                                                                                                                                                            | Ranked backlog of startup ideas    |
| **Market Validation**    | AutoGPT + browser/search plugins ([github.com](https://github.com/Significant-Gravitas/AutoGPT?utm_source=chatgpt.com))                                                                                                                                                                            | TAM/SAM analysis, sentiment scrape |
| **MVP Build**            | GPT‑Engineer CLI ([github.com](https://github.com/AntonOsika/gpt-engineer?utm_source=chatgpt.com)) + OpenDevin shell ([github.com](https://github.com/AI-App/OpenDevin.OpenDevin?utm_source=chatgpt.com))                                                                                          | Working repo with tests & CI       |
| **Launch & Growth**      | SuperAGI GTM stack ([superagi.com](https://superagi.com/open-source/?utm_source=chatgpt.com)) + Social‑GPT campaigns ([github.com](https://github.com/Social-GPT/agent?utm_source=chatgpt.com)) + Dittofeed journeys ([github.com](https://github.com/dittofeed/dittofeed?utm_source=chatgpt.com)) | Multi‑channel user acquisition     |
| **Analytics & Decision** | PostHog events & dashboards ([github.com](https://github.com/PostHog/posthog?utm_source=chatgpt.com))                                                                                                                                                                                              | KPI scorecard for stop/pivot       |

---

## System Architecture

```mermaid
flowchart TD
  A[Idea Queue] -->|pick| B(MetaGPT Ideator)
  B --> C[Validation Crew]
  C -->|pass| D{KPI Gate}
  D -- Good --> E[Build Crew (GPT‑Engineer & OpenDevin)]
  E --> F[Deploy → Fly.io]
  F --> G[Launch Crew (SuperAGI + Social‑GPT)]
  G --> H[Analytics (PostHog)]
  H --> D
  D -- Kill --> Z[Archive & Restart]
```

All agents communicate via an event bus (Redis/NATS) and maintain contextual memory in a vector store. LangGraph orchestrates stateful flows, while Flowise offers a drag‑and‑drop UI for rapid iteration.

---

## Repository Layout

```
agentic-startup-studio/
├── agents/            # Role definitions & SOPs
│   ├── ideation/
│   ├── build/
│   └── growth/
├── configs/
│   ├── langgraph/
│   └── superagi/
├── datasets/          # Vector DB exports
├── mvp_templates/     # Seed blueprints for GPT-Engineer
├── scripts/           # CLI utilities
├── docker-compose.yml # One‑command stack spin‑up
└── README.md          # <- you are here
```

---

## Quick Start

1. **Clone & bootstrap**

   ```bash
   git clone https://github.com/your-org/agentic-startup-studio.git
   cd agentic-startup-studio
   docker compose up -d  # spins up Redis, Postgres, LangGraph worker, PostHog, Dittofeed
   ```
2. **Install CLI deps**

   ```bash
   pip install metagpt crewai gpt-engineer superagi-langchain open-devin
   ```
3. **Seed first idea**

   ```bash
   python scripts/seed_idea.py "HIPAA compliance checker SaaS"
   ```

---

## Running an Experiment

```bash
make run-cycle MAX_TOKENS=100000 KPI_USERS=100
```

The `run-cycle` Make target executes:

1. Ideation & ranking
2. Validation scrape (search + sentiment)
3. MVP generation in `/workspace/<idea_slug>`
4. Containerised deploy (Fly.io default)
5. Launch campaign across X/Twitter, LinkedIn & Email
6. 72‑hour KPI watch; if goals unmet → archive.

All actions are logged to PostHog dashboards and can be replayed in Grafana/Tempo.

---

## Safety & Governance

* **Agbenchmark suite** ensures agents pass 23 tasks before being allowed to deploy code ([github.com](https://github.com/Significant-Gravitas/AutoGPT?utm_source=chatgpt.com)).
* **Veto Agent**: A read‑only watchdog monitors plans and can halt jobs that violate predefined policies.
* **Cost Guardrail**: Token usage capped per cycle via environment vars (`MAX_TOKENS`).
* **License Scanner**: SPDX check runs on third‑party code imported by agents during build.

---

## Roadmap

* [ ] Self‑hosted vector memory with pgvector
* [ ] Fine‑tuned small models for local draft reasoning
* [ ] Multi‑tenant mode (spin parallel idea tracks)
* [ ] UI dashboard (Next.js + shadcn) for non‑tech PMs

---

## Contributing

We welcome PRs for new agent roles, evaluation tasks, and deployment adapters. Please see `CONTRIBUTING.md` for code style and CI details.

---

## License

MIT for code; content under CC‑BY‑4.0.
