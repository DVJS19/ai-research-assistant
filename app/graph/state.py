"""
LangGraph shared state definition.
Every node reads from and writes to this TypedDict.

Phase 3 (single agent): uses topic, messages, tool_results, budget fields.
Phase 5 (multi-agent):  adds web_result, kb_result, synthesis fields.
"""

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # ── Core ──────────────────────────────────────────────────────────────────
    run_id: str
    topic: str

    # ── Agent conversation (ReAct loop) ───────────────────────────────────────
    # Annotated[list, operator.add] means each node APPENDS — never overwrites.
    # Safe for parallel workers writing to the same list.
    messages: Annotated[list[dict], operator.add]

    # ── Tool results (Phase 3: single agent) ─────────────────────────────────
    tool_results: dict[str, Any]  # keyed by tool name

    # ── Worker results (Phase 5: multi-agent) ────────────────────────────────
    web_result: dict | None
    kb_result: dict | None
    wiki_result: dict | None  # wikipedia worker

    # ── Synthesis output ──────────────────────────────────────────────────────
    report: dict | None
    confidence_scores: dict[str, float]

    # ── Errors (append-only — safe for parallel workers) ─────────────────────
    errors: Annotated[list[str], operator.add]

    # ── Budget tracking ───────────────────────────────────────────────────────
    step_count: int
    tokens_used: int
    cost_usd: float
