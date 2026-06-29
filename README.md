# AI Research Assistant

A multi-agent AI research system that takes a topic and produces a structured research report by searching the web, querying an internal knowledge base, and synthesising findings using LangGraph, LiteLLM, and Pinecone.

- Cross-referenced findings from web, internal KB, and Wikipedia
- Cited sources with URLs and document references  
- Confidence score (0-1) based on source agreement
- Cost tracking per run (~$0.04 average)
- Redis cache hit on repeat queries (~100ms vs ~60s)

## Architecture


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


## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent framework | LangGraph 1.x | StateGraph, Send API, checkpoint |
| LLM (workers) | Claude Haiku 4.5 | Fast, cost-efficient research |
| LLM (synthesis) | Claude Sonnet 4.6 | High-quality cross-reference |
| Vector DB | Pinecone | Hybrid search on knowledge base |
| Reranker | Cohere Rerank | Precision improvement on retrieval |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim vectors |
| Web search | Brave Search API | Current web results |
| Cache | Redis | Exact match cache (TTL 1hr) |
| Checkpoints | Postgres | LangGraph state + crash recovery |
| Observability | LangSmith + structlog | Node traces + structured logs |
| API | FastAPI | Async REST API |
| Package manager | UV | Fast dependency management |

## Performance

| Metric | Value |
|---|---|
| Avg cost per run | ~$0.04 |
| Cache hit response | < 100ms |
| Full run time | ~60 seconds |
| Knowledge base | 85 vectors across 7 PDFs |
| Golden set recall | 76% (9/10 questions) |
| Confidence threshold for cache | 0.75 |

## Quick start

```bash
# 1. Clone and setup
git clone https://github.com/DVJS19/ai-research-assistant
cd ai-research-assistant
bash scripts/setup.sh

# 2. Fill in API keys
cp .env.example .env
# Edit .env with your keys

# 3. Start services
docker compose up -d

# 4. Run
uv run uvicorn app.main:app --reload

# 5. Test
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
