from langgraph.types import Send

from app.graph.state import ResearchState
from app.logger import get_logger

log = get_logger(__name__)


async def orchestrator_node(state: ResearchState) -> dict:
    """
    Decomposes the research topic.
    Returns state update — fan-out happens via the conditional edge below.
    """
    topic = state["topic"]
    run_id = state["run_id"]

    log.info("orchestrator_dispatching", run_id=run_id, topic=topic[:60], workers=3)

    # Just return the state unchanged — the routing function does the fan-out
    return {}


def dispatch_workers(state: ResearchState) -> list[Send]:
    """
    Conditional edge function — returns Send objects to fan out to workers.
    Called by LangGraph after orchestrator_node completes.
    """
    topic = state["topic"]
    run_id = state["run_id"]

    return [
        Send("web_research", {"topic": topic, "run_id": run_id}),
        Send("kb_research", {"topic": topic, "run_id": run_id}),
        Send("wiki_research", {"topic": topic, "run_id": run_id}),
    ]
