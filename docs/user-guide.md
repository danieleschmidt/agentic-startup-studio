# Data Pipeline User Guide

This guide explains how to use the Agentic Startup Studio data pipeline for submitting, processing, and managing startup ideas. It covers the complete workflow from idea submission through automated deployment and testing.

## Table of Contents

- [Getting Started](#getting-started)
- [Understanding the Pipeline](#understanding-the-pipeline)
- [CLI Interface Usage](#cli-interface-usage)
- [Submitting Startup Ideas](#submitting-startup-ideas)
- [Quality Gates and Scoring](#quality-gates-and-scoring)
- [Monitoring Pipeline Progress](#monitoring-pipeline-progress)
- [Sample Startup Ideas](#sample-startup-ideas)
- [Understanding Reports and Metrics](#understanding-reports-and-metrics)
- [Best Practices](#best-practices)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Getting Started

### Prerequisites

Before using the pipeline, ensure you have:
- Valid API credentials (OpenAI, Google Ads, PostHog)
- Database access and proper environment setup
- Understanding of startup validation concepts
- Familiarity with basic command-line operations

### First-Time Setup

```bash
# Navigate to the project directory
cd /path/to/agentic-startup-studio

# Activate the Python environment
source venv/bin/activate

# Check system status
python -m pipeline.cli.ingestion_cli health

# View configuration
python -m pipeline.cli.ingestion_cli config
```

### Quick Start Example

```bash
# Submit your first idea
python -m pipeline.cli.ingestion_cli create \
  --title "AI-Powered Task Management" \
  --description "A smart task management app that uses AI to prioritize and schedule tasks based on user behavior and deadlines" \
  --category "productivity" \
  --problem "People struggle to prioritize tasks effectively" \
  --solution "AI analyzes patterns to suggest optimal task ordering"

# Check pipeline status
python -m pipeline.cli.ingestion_cli list --status=processing
```

---

## Understanding the Pipeline

### Pipeline Stages

The system processes ideas through six distinct stages:

| Stage | Purpose | Duration | Quality Gate |
|-------|---------|----------|--------------|
| **Ideate** | Initial capture and validation | 5-10 minutes | Input validation, duplicate detection |
| **Research** | Evidence collection via RAG | 30-60 minutes | Evidence score ≥0.7, ≥3 citations |
| **Deck** | Pitch deck generation | 15-30 minutes | Accessibility ≥90, content completeness |
| **Investors** | Multi-agent evaluation | 45-90 minutes | Composite score ≥0.8, consensus ≥70% |
| **SmokeTest** | Landing page and ads | 60-120 minutes | Campaign deployment, budget limits |
| **MVP** | Code generation and deployment | 2-4 hours | Test coverage ≥90%, health checks |

### Success Criteria

**Quality Metrics:**
- Evidence score: ≥0.7 (weighted: credibility 40%, relevance 30%, accessibility 20%, recency 10%)
- Investor consensus: ≥70% agreement between VC and Angel agents
- Smoke test conversion: >5% signup rate target
- Technical quality: ≥90% test coverage, Lighthouse score >90

**Budget Constraints:**
- Total cycle budget: ≤$62 (OpenAI $12, Google Ads $45, Infrastructure $5)
- Automatic shutdown at 100% budget utilization
- Warning alerts at 80%, critical alerts at 95%

---

## CLI Interface Usage

### Core Commands

**Creating Ideas:**
```bash
# Interactive creation
python -m pipeline.cli.ingestion_cli create

# Non-interactive with all parameters
python -m pipeline.cli.ingestion_cli create \
  --title "Smart Home Energy Optimizer" \
  --description "IoT system that optimizes home energy usage" \
  --category "energy" \
  --problem "High energy bills and waste" \
  --solution "ML-driven optimization algorithms" \
  --market "Homeowners with smart devices" \
  --evidence "https://energy.gov/stats,https://iot-market-report.com" \
  --force
```

**Listing and Filtering:**
```bash
# List all ideas
python -m pipeline.cli.ingestion_cli list

# Filter by status
python -m pipeline.cli.ingestion_cli list --status=validated

# Filter by stage and category
python -m pipeline.cli.ingestion_cli list \
  --stage=research \
  --category=technology \
  --limit=10

# Search with keywords
python -m pipeline.cli.ingestion_cli list --search="AI machine learning"

# JSON output for integration
python -m pipeline.cli.ingestion_cli list --output=json
```

**Idea Management:**
```bash
# View detailed information
python -m pipeline.cli.ingestion_cli show <idea-id>

# Update existing idea
python -m pipeline.cli.ingestion_cli update <idea-id> \
  --title "Updated Title" \
  --description "Enhanced description"

# Manually advance stage (if quality gates passed)
python -m pipeline.cli.ingestion_cli advance <idea-id> research

# Find similar ideas
python -m pipeline.cli.ingestion_cli similar <idea-id> --limit=5
```

### Output Formats

**Table Format (Default):**
```
AI-Powered Task Management
  ID: 123e4567-e89b-12d3-a456-426614174000
  Status: processing
  Stage: research
  Progress: ████████░░░░░░░░░░░░ 40.0%
  Created: 2025-01-15 14:30
```

**JSON Format:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "AI-Powered Task Management",
  "status": "processing",
  "stage": "research",
  "progress": 0.4,
  "created_at": "2025-01-15T14:30:00"
}
```

---

## Submitting Startup Ideas

### Required Fields

| Field | Description | Length | Example |
|-------|-------------|--------|---------|
| **Title** | Clear, descriptive name | 10-200 chars | "AI-Powered Task Management" |
| **Description** | Detailed explanation | 10-5000 chars | "A smart task management app..." |
| **Category** | Business domain | Enum | technology, healthcare, finance, etc. |

### Optional Fields

| Field | Description | Purpose |
|-------|-------------|---------|
| **Problem Statement** | What problem does this solve? | Research focus |
| **Solution Description** | How does your solution work? | Investor evaluation |
| **Target Market** | Who are your customers? | Market sizing |
| **Evidence Links** | Supporting research URLs | Evidence validation |

### Input Validation Rules

**Automatic Checks:**
- Title uniqueness (fuzzy matching at 70% similarity)
- Content length constraints (10-5000 characters)
- Profanity and spam detection
- HTML sanitization and security filtering
- Category validation against predefined taxonomy

**Quality Recommendations:**
- Include specific metrics and data in descriptions
- Provide credible evidence links (avoid paywalls)
- Use clear, jargon-free language
- Define the target market specifically
- Explain the competitive advantage

### Example Submissions

**Technology Startup:**
```bash
python -m pipeline.cli.ingestion_cli create \
  --title "CodeReview AI Assistant" \
  --description "An AI-powered tool that provides intelligent code review suggestions, identifies bugs, and recommends improvements using advanced static analysis and machine learning models trained on millions of code repositories" \
  --category "technology" \
  --problem "Manual code reviews are time-consuming and inconsistent" \
  --solution "AI analyzes code patterns and suggests improvements in real-time" \
  --market "Software development teams at companies with 10+ developers" \
  --evidence "https://github.com/reports/developer-productivity,https://stackoverflow.com/insights/developer-survey"
```

**Healthcare Startup:**
```bash
python -m pipeline.cli.ingestion_cli create \
  --title "Medication Adherence Tracker" \
  --description "Mobile app that helps patients track medication schedules, provides reminders, and connects with healthcare providers to monitor adherence and adjust treatments" \
  --category "healthcare" \
  --problem "50% of patients don't take medications as prescribed" \
  --solution "Smart reminders with provider integration and progress tracking" \
  --market "Chronic disease patients and healthcare providers" \
  --evidence "https://www.who.int/adherence-report,https://nejm.org/medication-compliance"
```

---

## Quality Gates and Scoring

### Research to Deck Quality Gate

**Requirements:**
- Evidence score ≥ 0.7
- Minimum 3 credible citations
- Source accessibility rate ≥ 80%
- Source diversity ≥ 2 different domains

**Scoring Algorithm:**
```
Evidence Score = (
  Credibility × 0.40 +
  Relevance × 0.30 +
  Accessibility × 0.20 +
  Recency × 0.10
)
```

**Common Failure Reasons:**
- Insufficient evidence sources (< 3 citations)
- Low-quality or unverified sources
- Paywall-blocked content (reduces accessibility)
- Outdated research (> 5 years old)

### Deck to Investor Quality Gate

**Requirements:**
- Lighthouse accessibility score ≥ 90
- Exactly 10 presentation slides
- Content completeness ≥ 80%
- Valid Marp syntax formatting

**Content Validation:**
- Executive summary present
- Market size analysis included
- Competitive landscape covered
- Financial projections provided
- Team information detailed

### Investor to Smoke Test Quality Gate

**Requirements:**
- Composite investor score ≥ 0.8 (configurable via `FUNDING_THRESHOLD`)
- Minimum 2 investor evaluations (VC + Angel agents)
- Consensus level ≥ 70% agreement
- No critical bias flags detected

**Investor Scoring Rubric:**
```
Investor Score = (
  Team Quality × 0.30 +
  Market Opportunity × 0.40 +
  Technical Moat × 0.20 +
  Evidence Quality × 0.10
)
```

**Bypass Options:**
- Research and Deck gates: Manual approval available
- Investor gate: **No bypass allowed** (funding decision)

---

## Monitoring Pipeline Progress

### Real-Time Status Tracking

```bash
# Monitor specific idea
watch -n 30 'python -m pipeline.cli.ingestion_cli show <idea-id>'

# Monitor all processing ideas
watch -n 60 'python -m pipeline.cli.ingestion_cli list --status=processing'

# Check budget status
python -m pipeline.cli.ingestion_cli budget status
```

### Progress Indicators

**Status Values:**
- `draft`: Idea created, pending validation
- `validating`: Input validation in progress
- `validated`: Passed initial validation
- `researching`: Evidence collection phase
- `building`: Deck generation phase
- `testing`: Investor evaluation phase
- `deployed`: Smoke test campaign active
- `completed`: Full pipeline success
- `rejected`: Failed quality gate or validation
- `archived`: Manually archived

**Progress Calculation:**
```
Progress = (Current Stage Number / Total Stages) × Stage Completion
Stage Completion = Task Progress within Current Stage
```

### Error Handling and Recovery

**Automatic Retries:**
- Transient API failures: 3 attempts with exponential backoff
- Network timeouts: 2 attempts with increased timeout
- Rate limiting: Automatic delay and retry

**Manual Intervention Required:**
- Quality gate failures beyond retry threshold
- Budget constraints exceeded
- External service authentication issues

---

## Sample Startup Ideas

### High-Quality Example

**Title:** "AI-Powered Personal Finance Coach"

**Description:** "A mobile application that analyzes users' spending patterns, income, and financial goals to provide personalized budgeting advice and investment recommendations. Uses machine learning to adapt suggestions based on user behavior and market conditions."

**Problem Statement:** "70% of Americans live paycheck to paycheck despite having average household incomes, indicating poor financial planning and budgeting skills."

**Solution Description:** "AI analyzes bank transactions, categorizes spending, identifies waste, and provides actionable recommendations with gamification elements to encourage better financial habits."

**Target Market:** "Working professionals aged 25-45 with household incomes $40k-$150k who want to improve their financial health but lack expertise."

**Evidence Links:**
- https://federalreserve.gov/publications/consumer-credit
- https://bankrate.com/personal-finance/financial-security-poll
- https://mint.com/blog/budgeting-statistics

**Expected Pipeline Results:**
- Evidence Score: 0.85 (high-quality government and financial sources)
- Investor Score: 0.82 (large addressable market, proven demand)
- Smoke Test CTR: 6.2% (strong value proposition)

### Medium-Quality Example

**Title:** "Smart Garden Monitoring System"

**Description:** "IoT sensors and mobile app for monitoring soil moisture, light levels, and plant health in home gardens with automated watering recommendations."

**Problem Statement:** "Home gardeners struggle to maintain optimal growing conditions and often over or under-water plants."

**Expected Pipeline Results:**
- Evidence Score: 0.72 (moderate evidence availability)
- Investor Score: 0.75 (niche market, hardware complexity)
- Smoke Test CTR: 4.1% (seasonal interest variability)

### Low-Quality Example (Will Likely Fail)

**Title:** "Social Media for Pets"

**Description:** "A social network where pet owners can create profiles for their pets and share photos."

**Problem Statement:** "Pet owners want to share photos of their pets."

**Expected Pipeline Results:**
- Evidence Score: 0.42 (weak problem validation)
- Investor Score: 0.51 (crowded market, unclear monetization)
- Status: Likely rejected at investor gate

---

## Understanding Reports and Metrics

### Pipeline Performance Metrics

**Throughput Metrics:**
- Ideas processed per week
- Average time per stage
- Completion rate by category
- Quality gate pass rates

**Quality Metrics:**
- Evidence score distribution
- Investor agreement rates
- Smoke test conversion rates
- MVP deployment success rates

**Budget Metrics:**
- Cost per completed idea
- Budget utilization by category
- ROI on successful deployments

### Generated Reports

**Evidence Collection Report:**
```json
{
  "idea_id": "123e4567-e89b-12d3-a456-426614174000",
  "evidence_summary": {
    "total_sources": 5,
    "credible_sources": 4,
    "accessibility_rate": 0.8,
    "average_recency": "2.3 years",
    "evidence_score": 0.74
  },
  "citations": [
    {
      "url": "https://example.com/research",
      "title": "Market Research Report 2024",
      "credibility": 0.85,
      "relevance": 0.92,
      "accessible": true
    }
  ]
}
```

**Investor Evaluation Report:**
```json
{
  "idea_id": "123e4567-e89b-12d3-a456-426614174000",
  "evaluations": [
    {
      "agent_type": "vc",
      "overall_score": 0.82,
      "team_score": 0.75,
      "market_score": 0.85,
      "tech_score": 0.80,
      "reasoning": "Strong market opportunity with proven demand..."
    },
    {
      "agent_type": "angel",
      "overall_score": 0.78,
      "team_score": 0.70,
      "market_score": 0.82,
      "tech_score": 0.75,
      "reasoning": "Good concept but execution risk remains..."
    }
  ],
  "consensus_score": 0.80,
  "agreement_level": 0.73,
  "funding_recommendation": "approved"
}
```

---

## Best Practices

### Idea Submission Guidelines

**Do:**
- Research existing solutions before submitting
- Provide specific, measurable problem statements
- Include credible evidence sources
- Use clear, professional language
- Define target market specifically
- Explain competitive advantages clearly

**Don't:**
- Submit overly broad or vague ideas
- Include unverified or speculative claims
- Use technical jargon without explanation
- Ignore existing market solutions
- Provide generic problem statements
- Submit without evidence validation

### Quality Optimization

**Improve Evidence Scores:**
- Use government and academic sources (.gov, .edu)
- Include recent industry reports (< 2 years)
- Provide multiple source types (research, surveys, case studies)
- Ensure all links are accessible without paywalls
- Cite specific statistics and data points

**Improve Investor Scores:**
- Demonstrate clear market demand with data
- Explain defensible competitive advantages
- Show realistic financial projections
- Address potential risks and mitigation
- Highlight team experience and expertise

### Pipeline Efficiency

**Reduce Processing Time:**
- Submit complete information initially
- Provide high-quality evidence links
- Use standard business terminology
- Follow description length guidelines
- Validate ideas internally before submission

**Monitor Budget Usage:**
- Check budget status before major operations
- Prioritize high-quality ideas to maximize ROI
- Use staging environment for testing
- Monitor external API costs regularly

---

## Troubleshooting Common Issues

### Submission Problems

**Error: "Title already exists"**
```bash
# Check for similar ideas
python -m pipeline.cli.ingestion_cli list --search="your title keywords"

# Use --force flag if truly different
python -m pipeline.cli.ingestion_cli create --force ...
```

**Error: "Description too short/long"**
- Minimum: 10 characters
- Maximum: 5000 characters
- Ensure meaningful content, not just padding

**Error: "Invalid category"**
```bash
# Check valid categories
python -m pipeline.cli.ingestion_cli create --help
# Valid: technology, healthcare, finance, education, energy, etc.
```

### Pipeline Failures

**Stuck in Research Stage:**
```bash
# Check evidence quality
python -m pipeline.cli.ingestion_cli show <idea-id> --output=json

# Common issues:
# - Inaccessible evidence links
# - Low-quality or irrelevant sources
# - Insufficient citation count
```

**Failed Investor Gate:**
```bash
# Review investor feedback
python -m pipeline.cli.ingestion_cli show <idea-id>

# Common issues:
# - Weak market validation
# - Unclear value proposition
# - Strong existing competition
# - Technical feasibility concerns
```

### Budget Issues

**Budget Warning/Critical Alerts:**
```bash
# Check current budget status
python -m pipeline.cli.ingestion_cli budget status

# Pause non-critical operations
# Wait for budget cycle reset
# Optimize expensive operations
```

**Emergency Shutdown:**
```bash
# Budget exceeded - manual intervention required
# Contact system administrator
# Review and approve budget increase
# Reset budget cycle if appropriate
```

---

For technical issues, refer to:
- [Operations Manual](operations-manual.md) for system administration
- [Deployment Guide](deployment-guide.md) for setup problems
- [API Documentation](api-documentation.md) for integration details

**Support:** Contact your system administrator for pipeline issues or budget concerns.