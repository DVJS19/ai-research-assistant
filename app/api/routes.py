import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graph import graph as graph_module
from app.graph.state import ResearchState
from app.logger import get_logger
from app.observability import get_trace_id, persist_run_metrics

log = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    topic: str
    namespace: str = "default"
    include_kb: bool = True


class ResearchResponse(BaseModel):
    run_id: str
    trace_id: str
    status: str
    answer: str
    sources: list[dict]
    steps_taken: int
    cost_usd: float
    errors: list[str]


@router.post("", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """
    Run the research agent on a topic.

    Returns the full research result synchronously.
    For long-running topics, consider adding a background task pattern.
    """
    if graph_module.graph_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not initialised. Check startup logs.",
        )

    run_id = str(uuid.uuid4())
    trace_id = get_trace_id()  # set by middleware for this request
    log.info(
        "research_request_received", run_id=run_id, trace_id=trace_id, topic=request.topic[:80]
    )

    # Initial state — LangGraph merges this with the TypedDict defaults
    initial_state: ResearchState = {
        "run_id": run_id,
        "topic": request.topic,
        "messages": [],
        "tool_results": {},
        "web_result": None,
        "kb_result": None,
        "wiki_result": None,
        "report": None,
        "confidence_scores": {},
        "errors": [],
        "step_count": 0,
        "tokens_used": 0,
        "cost_usd": 0.0,
    }

    # thread_id isolates this run's checkpoints from all other runs
    config = {"configurable": {"thread_id": run_id}}
    status = "completed"
    try:
        final_state = await graph_module.graph_instance.ainvoke(initial_state, config=config)
    except Exception as e:
        log.error("graph_invocation_failed", run_id=run_id, trace_id=trace_id, error=str(e))
        status = "failed"
        # Persist failure metrics before raising
        await persist_run_metrics(
            run_id=run_id,
            topic=request.topic,
            steps_taken=0,
            tokens_used=0,
            cost_usd=0.0,
            error_count=1,
            status="failed",
        )
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    # Extract the final answer — the last assistant message with no tool calls
    messages = final_state.get("messages", [])
    answer = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            answer = msg.get("content", "")
            break

    # Extract sources from tool result messages
    sources = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                content = json.loads(msg.get("content", "{}"))
                results = content.get("results", [])
                for r in results:
                    if r.get("url"):
                        sources.append({"url": r["url"], "title": r.get("title", "")})
                    elif r.get("doc_id"):
                        sources.append({"doc_id": r["doc_id"], "score": r.get("score", 0)})
            except (json.JSONDecodeError, AttributeError):
                continue
    steps_taken = final_state.get("step_count", 0)
    tokens_used = final_state.get("tokens_used", 0)
    cost_usd = final_state.get("cost_usd", 0.0)
    errors = final_state.get("errors", [])

    # Persist metrics — non-blocking on failure
    await persist_run_metrics(
        run_id=run_id,
        topic=request.topic,
        steps_taken=steps_taken,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        error_count=len(errors),
        status=status,
    )
    log.info(
        "research_request_completed",
        run_id=run_id,
        trace_id=trace_id,
        steps=final_state.get("step_count", 0),
        cost=final_state.get("cost_usd", 0.0),
    )

    return ResearchResponse(
        run_id=run_id,
        trace_id=trace_id,
        status=status,
        answer=answer or "No answer generated.",
        sources=sources,
        steps_taken=steps_taken,
        cost_usd=round(cost_usd, 6),
        errors=errors,
    )


@router.get("/metrics/summary")
async def metrics_summary():
    """
    Return aggregated run metrics from Postgres.
    Shows average cost, steps, and run count.
    """
    from app.observability import _pool

    if _pool is None:
        raise HTTPException(status_code=503, detail="Metrics pool not ready")

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*)                                            AS total_runs,
                ROUND(AVG(cost_usd)::numeric, 6)                   AS avg_cost_usd,
                ROUND(AVG(steps_taken)::numeric, 2)                AS avg_steps,
                ROUND(AVG(tokens_used)::numeric, 0)                AS avg_tokens,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END)   AS failed_runs
            FROM run_metrics
        """)

    return dict(row)
