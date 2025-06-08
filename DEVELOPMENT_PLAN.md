# Development Plan – Agentic Startup Studio (Founder × Investor Edition)

> **Goal:** Build, test, and deploy the enhanced founder→investor→smoke‑test workflow in 4 sprints (≈6 weeks) with automated quality gates.

## Milestone 0 – Repo & CI Bootstrap (Day 0‑2)

| Task                                                            | Owner Agent    | Output                              |
| --------------------------------------------------------------- | -------------- | ----------------------------------- |
| Fork base repo & create `dev` branch                            | *System Agent* | GitHub repo with branch protections |
| Docker Compose for Redis, Postgres/pgvector, PostHog, Dittofeed | *DevOps Agent* | `docker-compose.yml`                |
| GitHub Actions CI for lint + unit tests                         | *CI Agent*     | `.github/workflows/ci.yml`          |

## Milestone 1 – Core Data Structures (Day 3‑7)

| Task                                                                                                                           | Owner Agent           | Acceptance Criteria           |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------- | ----------------------------- |
| Define `Idea` Pyd([arxiv.org](https://arxiv.org/html/2409.04109v1?utm_source=chatgpt.com)), `evidence`, `deck_path`, `status`) | *CTO Agent*           | ✓ Pass mypy & unit tests      |
| Implement `idea_ledger` module using pgvector + SQLModel                                                                       | *Data Engineer Agent* | ✓ CRUD ops, 90% test coverage |
| CLI: `scripts/idea.py new/list/update`                                                                                         | *SRE Agent*           | ✓ Click command group         |

## Milestone 2 – Founder Crew & Evidence Collector (Week 2)

| Task                               | Owner Agent                                                                                                | Tools                                                                   |                  |               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------- | ------------- |
| YAML prompts for CEO, CTO, VP R\&D | *Prompt Engineer Agent*([github.com](https://github.com/FoundationAgents/MetaGPT?utm_source=chatgpt.com))n | Implement `EvidenceCollector` node that enforces ≥3 citations per claim | *Research Agent* | AutoGPT + RAG |
| Unit tests with mocked browser     | *QA Agent*                                                                                                 | PyTest + vcrpy                                                          |                  |               |

## Milestone 3 – Pitch Loop (Week 3‑4)

| Task                                                                       | Owner Agent      | Details                  |
| -------------------------------------------------------------------------- | ---------------- | ------------------------ |
| `pitch_loop.py` LangGraph state machine (`Ideate→Research→Deck→Investors`) | *Workflow Agent* | Passes integration tests |
| Marp deck generator (`deck.marp`) from YAML spec                           | *Docs Agent*     | 10‑slide template        |
| Investor agents (VC, Angel) with scoring rubric YAML                       | *VC Agent*       | Uses Gemini 2.5 Pro      |
| Funding threshold configurable (`FUND_THRESHOLD`)                          | *Config Agent*   | env var                  |

## Milestone 4 – Smoke Test Automation (Week 4‑5)

| Task                                                                 | Owner Agent      | KPI                                                                                                                                          |
| -------------------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Static Next.js landing page template w/ Buy/Signup CTA               | *Frontend Agent* | Lighthouse > 90                                                                                                                              |
| Unbounce / Google Ads API integration                                | *Growth Agent*   | Deploy & budget cap                                                                                                                          |
| `run_smoke_test.py` script to deploy & pull metrics to PostHog       | *Growth Agent*   | CTR, conversion stored i([arxiv.org](https://arxiv.org/abs/2501.14844?utm_source=chatgpt.com))lestone 5 – Build Cycle Integration (Week 5‑6) |
| Task                                                                 | Owner Agent      | Dependencies                                                                                                                                 |
| ------                                                               | -------------    | --------------                                                                                                                               |
| Extend `make build-cycle` to fetch idea → GPT‑Engineer scaffold repo | *Pipeline Agent* | uses idea\_id                                                                                                                                |
| OpenDevin debug loop integration                                     | *Dev Agent*      | Passes unit tests                                                                                                                            |
| Launch via Fly.io; Livelihood health check                           | *DevOps Agent*   | 99% uptime on test                                                                                                                           |

## Milestone 6 – Monitoring & Cost Controls (Continuous)

* **Token Budget Sentinel**: alerts when run exceeds `MAX_TOKENS`.
* **Ad Budget Sentinel**: halts smoke test at \$50 spend.
* **Bias Monitor**: ensemble critics flag excessive sentiment skew.

## Deliverables & Success Metrics

\| Deliverable | Metric |
\|-----([deckmatch.com](https://www.deckmatch.com/?utm_source=chatgpt.com))n| Accepted Ideas / Month | ≥ 4 with smoke‑test signup > 5 % |
\| False‑Positive Rate (ideas passing smoke test but later ki([linkedin.com](https://www.linkedin.com/pulse/building-smoke-test-validate-your-business-ideas-quickly-haggas-lbn4e?utm_source=chatgpt.com)) CI Pass Rate | 100 % on `main` |
\| Infra Cost per Idea Cycle | ≤ \$12 GPT + \$50 ads |

## Toolchain Versions

* Python 3.11, Poetry 1.7
* GPT‑Engineer 0.4.3
* MetaGPT 0.7.x
* LangGraph 0.3.x
* CrewAI 0.2.x
* OpenDevin preview‑2025‑05
  ([arxiv.org](https://arxiv.org/pdf/2410.20024?utm_source=chatgpt.com))‑hosted
* pgvector 0.6

## Appendix – Rubric YAML Snippet

```yaml
rubric:
  team: {weight: 0.3, descripti([arxiv.org](https://arxiv.org/html/2404.02650v1?utm_source=chatgpt.com))ertise‑fit"}
  market: {weight: 0.4, description: "TAM/SAM quality, growth"}
  tech_m([openai.com](https://openai.com/api/pricing/?utm_source=chatgpt.com))description: "Defensible advantage"([github.com](https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks?utm_source=chatgpt.com)): {weight: 0.1, description: "Citation quality"}
thresholds:
  fund_score: 0.8
  smoke_ctr: 0.05
  smoke_signups: 100
```

---

> **Next Action:** run `make bootstrap` and complete Milestone 0. Automated agents will then pull this plan and create the necessary PRs.
