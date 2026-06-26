"""
Phase 1 tests — verify environment setup is correct.
Run: uv run pytest tests/test_config.py -v
"""

import pytest

from app.config import settings
from app.tools.registry import TOOL_REGISTRY, get_tool_schemas


def test_settings_load():
    """Settings object is created without errors."""
    assert settings is not None
    assert isinstance(settings.agent_max_steps, int)
    assert settings.agent_max_steps > 0


def test_budget_defaults():
    """Budget limits have sensible defaults."""
    assert settings.agent_max_steps == 15
    assert settings.agent_max_tokens == 50_000
    assert settings.agent_max_cost_usd == 2.00
    assert settings.agent_max_seconds == 300


def test_model_defaults():
    """Model names are set."""
    assert "gpt" in settings.worker_model or "claude" in settings.worker_model
    assert settings.embedding_model == "text-embedding-3-small"


def test_tool_registry_has_required_tools():
    """Tool registry contains the tools we need."""
    assert "web_search" in TOOL_REGISTRY
    assert "rag_search" in TOOL_REGISTRY


def test_tool_schemas_valid():
    """Tool schemas are in OpenAI function-calling format."""
    schemas = get_tool_schemas()
    assert len(schemas) >= 2
    for schema in schemas:
        assert "type" in schema
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_budget_governor_raises_on_exceeded():
    """Budget governor raises BudgetExceeded when limits hit."""
    from app.graph.budget import BudgetExceeded, check_budget
    from app.graph.state import ResearchState

    # Build a state that exceeds max_steps
    state: ResearchState = {
        "run_id": "test-run",
        "topic": "test",
        "messages": [],
        "tool_results": {},
        "web_result": None,
        "kb_result": None,
        "wiki_result": None,
        "report": None,
        "confidence_scores": {},
        "errors": [],
        "step_count": 999,  # exceeds limit
        "tokens_used": 0,
        "cost_usd": 0.0,
    }
    with pytest.raises(BudgetExceeded) as exc:
        check_budget(state)
    assert "max_steps" in exc.value.reason
