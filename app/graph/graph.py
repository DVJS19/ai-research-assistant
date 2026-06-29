
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.graph.nodes.kb_research import kb_research_node
from app.graph.nodes.orchestrator import dispatch_workers, orchestrator_node
from app.graph.nodes.synthesis import synthesis_node
from app.graph.nodes.web_research import web_research_node
from app.graph.nodes.wiki_research import wiki_research_node
from app.graph.state import ResearchState
from app.logger import get_logger

log = get_logger(__name__)

# Module-level instances — set once at startup, used across all requests
graph_instance = None
_pool: AsyncConnectionPool = None  # holds the connection pool open for the app lifetime


async def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Uses a persistent AsyncConnectionPool so the Postgres connection
    stays open across all requests — not just during graph build.

    Phase 3 (single agent):  START → research_node → END
    Phase 5 (multi-agent):
        START
          → orchestrator_node          (decomposes + Send API fan-out)
          → [web_research, kb_research, wiki_research]  (parallel)
          → synthesis_node             (cross-reference + confidence)
          → END

    Postgres checkpoint saver persists state after every node.
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

    # Add all nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("web_research", web_research_node)
    builder.add_node("kb_research", kb_research_node)
    builder.add_node("wiki_research", wiki_research_node)
    builder.add_node("synthesis", synthesis_node)

    # Entry point
    builder.add_edge(START, "orchestrator")

    # Fan-out via conditional edge — this is how Send works in newer LangGraph
    builder.add_conditional_edges(
        "orchestrator",
        dispatch_workers,
        ["web_research", "kb_research", "wiki_research"],
    )
    # Workers converge at synthesis
    builder.add_edge("web_research", "synthesis")
    builder.add_edge("kb_research", "synthesis")
    builder.add_edge("wiki_research", "synthesis")

    # Synthesis → END
    builder.add_edge("synthesis", END)

    # ── Checkpoint saver ───────────────────────────────────────────────────────
    checkpointer = AsyncPostgresSaver(conn=_pool)
    await checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)
    log.info(
        "graph_compiled",
        nodes=["orchestrator", "web_research", "kb_research", "wiki_research", "synthesis"],
    )
    return graph


async def close_pool():
    """Call at app shutdown to cleanly close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        log.info("postgres_pool_closed")
