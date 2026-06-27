import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graph import graph as graph_module
from app.graph.state import ResearchState
from app.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    topic: str
    namespace: str = "default"
    include_kb: bool = True


class ResearchResponse(BaseModel):
    run_id: str
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
    log.info("research_request_received", run_id=run_id, topic=request.topic[:80])

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

    try:
        final_state = await graph_module.graph_instance.ainvoke(initial_state, config=config)
    except Exception as e:
        log.error("graph_invocation_failed", run_id=run_id, error=str(e))
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
            import json

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

    log.info(
        "research_request_completed",
        run_id=run_id,
        steps=final_state.get("step_count", 0),
        cost=final_state.get("cost_usd", 0.0),
    )

    return ResearchResponse(
        run_id=run_id,
        status="completed",
        answer=answer or "No answer generated.",
        sources=sources,
        steps_taken=final_state.get("step_count", 0),
        cost_usd=round(final_state.get("cost_usd", 0.0), 6),
        errors=final_state.get("errors", []),
    )
