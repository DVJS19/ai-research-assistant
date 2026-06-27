import contextvars
import uuid

import asyncpg

from app.logger import get_logger

log = get_logger(__name__)

# ── Trace ID context variable ──────────────────────────────────────────────────
# One trace_id per agent run, stored in async context so concurrent runs
# never share or overwrite each other's trace_id.
# Usage:
#   trace_id_var.set("abc-123")   # set at start of request
#   trace_id_var.get()            # read anywhere in the same async context
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="no-trace")


def set_trace_id(trace_id: str) -> None:
    """Set the trace_id for the current async context."""
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """Get the trace_id for the current async context."""
    return _trace_id_var.get()


def new_trace_id() -> str:
    """Generate a new trace_id, set it, and return it."""
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    return trace_id


# ── Run metrics persistence ────────────────────────────────────────────────────
# Persists one row per agent run to Postgres.
# Lets you query: SELECT AVG(cost_usd), AVG(steps_taken) FROM run_metrics
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS run_metrics (
    id           SERIAL PRIMARY KEY,
    run_id       TEXT        NOT NULL,
    trace_id     TEXT        NOT NULL,
    topic        TEXT        NOT NULL,
    steps_taken  INTEGER     NOT NULL DEFAULT 0,
    tokens_used  INTEGER     NOT NULL DEFAULT 0,
    cost_usd     NUMERIC(10,6) NOT NULL DEFAULT 0,
    error_count  INTEGER     NOT NULL DEFAULT 0,
    status       TEXT        NOT NULL DEFAULT 'completed',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Module-level connection pool — set during app startup
_pool: asyncpg.Pool | None = None


async def setup_metrics_table(database_url: str) -> None:
    """
    Create the run_metrics table if it doesn't exist.
    Called once at app startup.
    Uses a separate asyncpg pool — not the psycopg pool used by LangGraph.
    """
    global _pool

    # asyncpg uses a different URL scheme than psycopg
    asyncpg_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    _pool = await asyncpg.create_pool(asyncpg_url, min_size=1, max_size=3)

    async with _pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)

    log.info("run_metrics_table_ready")


async def close_metrics_pool() -> None:
    """Close the metrics pool at app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        log.info("metrics_pool_closed")


async def persist_run_metrics(
    run_id: str,
    topic: str,
    steps_taken: int,
    tokens_used: int,
    cost_usd: float,
    error_count: int,
    status: str = "completed",
) -> None:
    """
    Persist one row of run metrics to Postgres.
    Called after every agent run completes (success or failure).
    Non-blocking on failure — observability should never break the app.
    """
    if _pool is None:
        log.warning("metrics_pool_not_ready", run_id=run_id)
        return

    trace_id = get_trace_id()

    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO run_metrics
                    (run_id, trace_id, topic, steps_taken, tokens_used,
                     cost_usd, error_count, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                run_id,
                trace_id,
                topic,
                steps_taken,
                tokens_used,
                cost_usd,
                error_count,
                status,
            )
        log.info("run_metrics_persisted", run_id=run_id, cost_usd=cost_usd, steps=steps_taken)
    except Exception as e:
        # Never let observability failures crash the app
        log.error("run_metrics_persist_failed", run_id=run_id, error=str(e))
