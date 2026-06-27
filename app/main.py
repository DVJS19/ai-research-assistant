"""
FastAPI entry point.
Phase 1: health check + placeholder /research endpoint.
Phase 3: /research calls the LangGraph agent.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.routes import router as research_router
from app.config import settings
from app.graph import graph as graph_module
from app.logger import get_logger, setup_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    setup_logging()

    # Explicitly push API keys into os.environ so LiteLLM can find them.
    # pydantic-settings loads .env into settings but not into os.environ automatically.
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    log.info("startup", env=settings.app_env, tracing=settings.langchain_tracing_v2)
    # Build and cache the LangGraph graph instance.
    # Done once at startup — shared across all requests.
    graph_module.graph_instance = await graph_module.build_graph()
    log.info("graph_ready")

    yield
    await graph_module.close_pool()
    log.info("shutdown")


app = FastAPI(
    title="AI Research Assistant",
    description="Multi-agent research system — LangGraph + LiteLLM + Pinecone",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ____ Routes _____________________________________________________________
app.include_router(research_router)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "env": settings.app_env,
        "graph": "ready" if graph_module.graph_instance else "not initialised",
    }


# ── Research endpoint (Phase 1: placeholder) ──────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str
    max_sources: int = 5
    include_kb: bool = True


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    message: str


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Phase 1: returns a placeholder.
    Phase 3: will invoke the LangGraph agent and return run_id.
    """
    import uuid

    run_id = str(uuid.uuid4())
    log.info("research_requested", run_id=run_id, topic=request.topic)

    # TODO Phase 3: invoke LangGraph graph here
    return ResearchResponse(
        run_id=run_id,
        status="pending",
        message=f"Research job queued for: {request.topic}. Full agent coming in Phase 3.",
    )


# ── Run directly (dev only) ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
