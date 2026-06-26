"""
Tool registry — the single source of truth for every tool the agent can call.
The dispatcher validates every LLM-generated tool call against this before execution.

Phase 3: web_search + rag_search + summarise
Phase 5: each worker agent gets its own subset
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    name: str
    description: str  # LLM uses this to decide when to call
    handler: Callable
    schema: dict  # JSON Schema for parameter validation
    idempotent: bool = True  # safe to retry?
    timeout_ms: int = 10_000


# Import handlers (added as each phase is built)
# from app.tools.web_search import web_search_handler
# from app.tools.rag_search import rag_search_handler


def _placeholder_handler(**kwargs: Any) -> dict:
    """Placeholder until real handlers are implemented in Phase 3."""
    return {"status": "placeholder", "kwargs": kwargs}


TOOL_REGISTRY: dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description=(
            "Search the web for recent information about a topic. "
            "Use for current events, news, product updates, and public information. "
            "Do NOT use for internal company documents — use rag_search instead."
        ),
        handler=_placeholder_handler,  # replaced in Phase 3
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Be specific. Max 100 characters.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to return. Max 10.",
                },
            },
            "required": ["query"],
        },
        idempotent=True,
        timeout_ms=8_000,
    ),
    "rag_search": Tool(
        name="rag_search",
        description=(
            "Search the internal knowledge base using semantic similarity. "
            "Use for internal documents, past research, company policies, and domain knowledge. "
            "Returns the most relevant document chunks with confidence scores."
        ),
        handler=_placeholder_handler,  # replaced in Phase 3
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query."},
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of chunks to return after reranking.",
                },
            },
            "required": ["query"],
        },
        idempotent=True,
        timeout_ms=3_000,
    ),
}


def get_tool(name: str) -> Tool | None:
    """Look up a tool by name. Returns None if not found."""
    return TOOL_REGISTRY.get(name)


def get_tool_schemas() -> list[dict]:
    """Return all tool schemas in OpenAI function-calling format for the LLM prompt."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        }
        for t in TOOL_REGISTRY.values()
    ]
