# ARAG: Adaptive Technical Interviewing Using Agentic RAG

A multi-agent system for adaptive technical interviewing powered by Agentic Retrieval-Augmented Generation. The system detects knowledge gaps in real-time, generates grounded follow-up questions, and produces evidence-based assessments — all backed by a domain knowledge base rather than LLM parametric memory.

## Architecture

```
Candidate ↔ DMA (Dialogue Manager) ↔ Specialized Agents ↔ Domain KB
```

| Agent | Role |
|-------|------|
| **DMA** | Session orchestration, intent classification, flow control (hybrid: deterministic + LLM) |
| **KGDA** | Detects knowledge gaps by comparing responses against retrieved KB properties |
| **QGA** | Generates grounded questions targeting identified gaps |
| **RAA** | Scores responses against retrieved ground truth with element-level auditability |
| **FSA** | Synthesizes end-of-session feedback with KB citations |

## Tech Stack

- **LLM**: GPT-4o (agents) / Claude 3.5 Sonnet (candidate simulation)
- **Vector Store**: ChromaDB
- **Embeddings**: text-embedding-3-small
- **Orchestration**: LangGraph
- **Evaluation**: RAGAS + custom metrics

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.11+

## Setup

```bash
# Install dependencies
uv sync --all-extras

# Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and ANTHROPIC_API_KEY

# Seed the knowledge base
uv run python scripts/seed_kb.py
```

## Usage

### Interactive Interview

```bash
uv run python scripts/run_interview.py
```

### Run Evaluation

```bash
# Full evaluation (4 variants × 2 persona levels × 3 repetitions)
uv run python scripts/run_evaluation.py

# Baseline and ablation experiments only
uv run python scripts/run_baseline.py --variants no_rag no_kgda no_raa_ground
```

## Project Structure

```
src/
├── config.py              # Settings (YAML + env vars)
├── models.py              # Shared data models (GapReport, AssessmentRecord, etc.)
├── kb/                    # Knowledge base (schema, loader, ChromaDB retriever)
├── agents/                # DMA, KGDA, QGA, RAA, FSA
├── prompts/               # Prompt templates per agent
├── graph/                 # LangGraph state, nodes, and builder
├── simulation/            # Synthetic candidate personas and simulator
├── evaluation/            # Metrics, RAGAS adapter, evaluation runner
└── baseline/              # Non-RAG baseline and ablation variants
```

## Knowledge Base

35 DSA concepts across three difficulty tiers (beginner/intermediate/advanced) stored in `data/kb/concepts.json`. Each entry includes:

- Definition and key properties
- Common misconceptions
- Example correct response
- Related concepts

## Evaluation

| Metric | RQ | Target |
|--------|-----|--------|
| Gap Coverage Rate | RQ1 | ≥ 0.80 |
| Redundancy Rate | RQ1 | ≤ 0.10 |
| Element Precision/Recall | RQ2 | — |
| Hallucination Rate | RQ2 | — |
| Evidence Citation Rate | RQ3 | — |
| RAGAS Faithfulness | RQ2/RQ3 | — |

## Testing

```bash
uv run pytest
```

## References

See [PROPOSAL.md](PROPOSAL.md) for the full research proposal and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the detailed implementation plan.
