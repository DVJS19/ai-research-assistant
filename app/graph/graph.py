from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.graph.nodes.research import research_node
from app.graph.state import ResearchState
from app.logger import get_logger

log = get_logger(__name__)

# Module-level instances — set once at startup, used across all requests
graph_instance = None
_pool = None  # holds the connection pool open for the app lifetime


async def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Uses a persistent AsyncConnectionPool so the Postgres connection
    stays open across all requests — not just during graph build.

    Phase 3 (single agent):  START → research_node → END
    Phase 5 (multi-agent):   START → orchestrator → workers → synthesis → END
    """
    global _pool

    # ── Open a persistent connection pool ─────────────────────────────────────
    # min_size=1 keeps at least one connection alive at all times.
    # The pool stays open until close_pool() is called at shutdown.
    conn_string = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    _pool = AsyncConnectionPool(
        conninfo=conn_string,
        min_size=1,
        max_size=5,
        open=False,  # we open it explicitly below
    )
    await _pool.open()
    log.info("postgres_pool_opened")

    # ── Create checkpointer using the persistent pool ─────────────────────────
    checkpointer = AsyncPostgresSaver(conn=_pool)
    await checkpointer.setup()

    # ── Define graph structure ─────────────────────────────────────────────────
    builder = StateGraph(ResearchState)

    builder.add_node("research", research_node)
    builder.add_edge(START, "research")
    builder.add_edge("research", END)

    # ── Compile ────────────────────────────────────────────────────────────────
    graph = builder.compile(checkpointer=checkpointer)
    log.info("graph_compiled", nodes=["research"])
    return graph


async def close_pool():
    """Call at app shutdown to cleanly close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        log.info("postgres_pool_closed")
