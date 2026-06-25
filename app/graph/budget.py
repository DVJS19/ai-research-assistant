"""
Budget governor — checked at the start of every LangGraph node.
Prevents runaway cost and infinite loops.

Usage:
    from app.graph.budget import check_budget, BudgetExceeded
    check_budget(state)   # raises BudgetExceeded if any limit hit
"""
from enum import Enum
from app.config import settings
from app.graph.state import ResearchState
from app.logger import get_logger

log = get_logger(__name__)


class BudgetExceeded(Exception):
    """Raised when any budget limit is hit."""
    def __init__(self, reason: str, state: ResearchState):
        self.reason = reason
        self.state  = state
        super().__init__(f"Budget exceeded: {reason}")


def check_budget(state: ResearchState) -> None:
    """
    Call at the start of every node.
    Raises BudgetExceeded if any limit is hit.
    Also emits a warning log at 80% of each limit.
    """
    run_id = state.get("run_id", "unknown")

    # ── Hard limits ───────────────────────────────────────────────────────────
    if state.get("step_count", 0) >= settings.agent_max_steps:
        log.warning("budget_steps_exceeded", run_id=run_id,
                    steps=state["step_count"], limit=settings.agent_max_steps)
        raise BudgetExceeded("max_steps", state)

    if state.get("tokens_used", 0) >= settings.agent_max_tokens:
        log.warning("budget_tokens_exceeded", run_id=run_id,
                    tokens=state["tokens_used"], limit=settings.agent_max_tokens)
        raise BudgetExceeded("max_tokens", state)

    if state.get("cost_usd", 0.0) >= settings.agent_max_cost_usd:
        log.warning("budget_cost_exceeded", run_id=run_id,
                    cost=state["cost_usd"], limit=settings.agent_max_cost_usd)
        raise BudgetExceeded("max_cost_usd", state)

    # ── 80% warnings ─────────────────────────────────────────────────────────
    if state.get("cost_usd", 0.0) >= settings.agent_max_cost_usd * 0.8:
        log.warning("budget_80pct_warning", run_id=run_id,
                    cost=state["cost_usd"], limit=settings.agent_max_cost_usd)
