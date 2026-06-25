# AI Research Assistant

A multi-agent AI research system that takes a topic and produces a structured research report by searching the web, querying an internal knowledge base, and synthesising findings using LangGraph, LiteLLM, and Pinecone.

## Architecture

```
POST /research
      ↓
  FastAPI + ALB
      ↓
  LangGraph StateGraph
  ┌─────────────────────────────────────────┐
  │  START → orchestrator_node              │
  │              ↓ Send API (parallel)      │
  │  ┌──────────┬──────────┬─────────────┐  │
  │  │ web_node │ kb_node  │  wiki_node  │  │
  │  └──────────┴──────────┴─────────────┘  │
  │              ↓ merge state              │
  │         synthesis_node (GPT-4o)         │
  │              ↓ conditional              │
  │     hitl_node ← conf < 0.75            │
  │              ↓                          │
  │          output_node                    │
  │              ↓                          │
  │             END                         │
  └─────────────────────────────────────────┘
      ↓
  JSON report + PDF (S3)
```

## Tech stack

| Component | Tool | Purpose |
|---|---|---|
| Agent framework | LangGraph | StateGraph, Send API, checkpoint |
| LLM gateway | LiteLLM | Model routing, circuit breaker, cost tracking |
| LLM (workers) | GPT-4o-mini | Cost-efficient research tasks |
| LLM (synthesis) | GPT-4o | High-quality cross-reference and report |
| Vector DB | Pinecone | Hybrid search on knowledge base |
| Reranker | Cohere Rerank | Precision improvement on retrieval |
| Checkpoints | Postgres (RDS) | LangGraph state persistence + crash recovery |
| Cache | Redis | Prompt cache + hot state |
| Observability | LangSmith + structlog | Node traces + structured logs |
| API | FastAPI | Async REST API |
| Package manager | UV | Fast dependency management |

## Quick start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/ai-research-assistant
cd ai-research-assistant
bash scripts/setup.sh

# 2. Fill in API keys
vi .env

# 3. Run
uv run uvicorn app.main:app --reload

# 4. Test
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "LangGraph multi-agent patterns"}'
```

## Prerequisites

- Python 3.11+
- Docker (for local Postgres + Redis)
- UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- API keys: OpenAI, Pinecone, Cohere, Brave Search, LangSmith (optional)

## Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint + format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy app/
```

## Eval

```bash
# Run golden set evaluation (Phase 4+)
uv run python scripts/run_eval.py
```

Results are saved to `evals/results/` with per-question scores and overall recall@5.

## Cost

Average cost per research run: **~$0.04** (GPT-4o-mini workers + GPT-4o synthesis).  
Budget ceiling per run: **$2.00** (50× headroom).

---

*Built as a portfolio project demonstrating production LangGraph + RAG patterns.*
