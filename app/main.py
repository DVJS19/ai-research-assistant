"""
FastAPI entry point.
health check + placeholder /research endpoint.
/research calls the LangGraph agent.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as research_router
from app.cache import close_cache, setup_cache
from app.config import settings
from app.graph import graph as graph_module
from app.logger import get_logger, setup_logging
from app.observability import (
    close_metrics_pool,
    new_trace_id,
    setup_metrics_table,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    setup_logging()
    log.info("startup", env=settings.app_env)

    # pydantic-settings loads .env into settings but not into os.environ automatically.
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    log.info("startup", env=settings.app_env, tracing=settings.langchain_tracing_v2)
    # Build and cache the LangGraph graph instance.
    # Done once at startup — shared across all requests.
    graph_module.graph_instance = await graph_module.build_graph()
    log.info("graph_ready")

    # Create run_metrics table if it doesn't exist
    await setup_metrics_table(settings.database_url)
    try:
        await setup_cache()
    except Exception as e:
        log.warning("cache_setup_failed", error=str(e))
        log.warning("continuing_without_cache")

    yield

    await graph_module.close_pool()
    await close_metrics_pool()
    await close_cache()
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


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """
    Assign a trace_id to every incoming request.
    The trace_id is set in contextvars so every log event
    in this request can include it for correlation.
    """
    trace_id = new_trace_id()
    # Attach to request state so routes can access it if needed
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


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
