from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.state import ResearchState


# ── Dispatcher tests ───────────────────────────────────────────────────────────
class TestDispatcher:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Dispatcher returns structured error for hallucinated tool names."""
        from app.tools.dispatcher import dispatch

        result = await dispatch("hallucinated_tool", {"query": "test"})

        assert "error" in result
        assert "hallucinated_tool" in result["error"]
        assert "available" in result

    @pytest.mark.asyncio
    async def test_invalid_params_returns_error(self):
        """Dispatcher returns structured error when required params are missing."""
        from app.tools.dispatcher import dispatch

        # web_search requires 'query' — omitting it should fail validation
        result = await dispatch("web_search", {})

        assert "error" in result
        assert "Invalid parameters" in result["error"]

    @pytest.mark.asyncio
    @patch("app.tools.web_search.httpx.AsyncClient")
    async def test_web_search_dispatches_correctly(self, mock_client_cls):
        """Valid web_search call reaches the handler and returns results."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "LangGraph Overview",
                        "url": "https://example.com/langgraph",
                        "description": "LangGraph is a framework for building agents.",
                    }
                ]
            }
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.tools.dispatcher import dispatch

        result = await dispatch("web_search", {"query": "LangGraph agents"})

        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["url"] == "https://example.com/langgraph"


# ── Research node tests ────────────────────────────────────────────────────────
class TestResearchNode:
    def _make_state(self, topic: str = "LangGraph") -> ResearchState:
        return {
            "run_id": "test-run-001",
            "topic": topic,
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

    def _make_anthropic_response(
        self,
        tool_use_blocks: list | None = None,
        text_content: str = "",
        stop_reason: str | None = None,
    ):
        """Build a fake Anthropic messages.create response."""
        content_blocks = []

        if text_content:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = text_content
            content_blocks.append(text_block)

        if tool_use_blocks:
            content_blocks.extend(tool_use_blocks)
            resolved_stop_reason = stop_reason or "tool_use"
        else:
            resolved_stop_reason = stop_reason or "end_turn"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 100

        mock_response = MagicMock()
        mock_response.content = content_blocks
        mock_response.usage = mock_usage
        mock_response.stop_reason = resolved_stop_reason
        return mock_response

    def _make_tool_use_block(self, name: str, arguments: dict, block_id: str = "toolu_abc123"):
        """Build a fake Anthropic tool_use content block."""
        block = MagicMock()
        block.type = "tool_use"
        block.id = block_id
        block.name = name
        block.input = arguments  # Anthropic passes input as a dict, not JSON string
        return block

    @pytest.mark.asyncio
    @patch("app.graph.nodes.research.anthropic.AsyncAnthropic")
    @patch("app.tools.web_search.httpx.AsyncClient")
    async def test_research_node_one_tool_call_then_finish(
        self,
        mock_http_cls,
        mock_anthropic_cls,
    ):
        """
        Simulates a full ReAct loop:
        Iteration 1: LLM calls web_search
        Iteration 2: LLM outputs final answer (no tool calls) → FINISH
        """
        # Fake web search HTTP response
        mock_http_response = MagicMock()
        mock_http_response.raise_for_status = MagicMock()
        mock_http_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Test", "url": "https://test.com", "description": "Test result"}
                ]
            }
        }
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_http_response)
        mock_http_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        # Iteration 1: LLM decides to call web_search
        tool_block = self._make_tool_use_block("web_search", {"query": "LangGraph"})
        response_1 = self._make_anthropic_response(tool_use_blocks=[tool_block])

        # Iteration 2: LLM outputs final answer — no tool calls = FINISH
        response_2 = self._make_anthropic_response(
            text_content="LangGraph is a framework for stateful multi-actor LLM applications.",
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=[response_1, response_2])
        mock_anthropic_cls.return_value = mock_client

        from app.graph.nodes.research import research_node

        result = await research_node(self._make_state())

        assert result["step_count"] == 2

        messages = result["messages"]
        roles = [m["role"] for m in messages]
        assert "tool" in roles
        assert "assistant" in roles

        final_assistant = [
            m for m in messages if m["role"] == "assistant" and not m.get("tool_calls")
        ]
        assert len(final_assistant) >= 1
        assert "LangGraph" in final_assistant[-1]["content"]

    @pytest.mark.asyncio
    @patch("app.graph.nodes.research.anthropic.AsyncAnthropic")
    async def test_budget_exceeded_returns_partial(self, mock_anthropic_cls):
        """When budget is exceeded, node returns partial result not exception."""
        from app.graph.nodes.research import research_node

        state = self._make_state()
        state["step_count"] = 999  # over the limit of 15

        result = await research_node(state)

        assert "errors" in result
        assert any("budget" in e.lower() for e in result["errors"])

        # LLM should NOT have been called — budget check happens first
        mock_anthropic_cls.return_value.messages.create.assert_not_called()
