import json

import anthropic

from app.config import settings
from app.graph.budget import BudgetExceeded, check_budget
from app.graph.state import ResearchState
from app.logger import get_logger
from app.tools.dispatcher import dispatch
from app.tools.registry import get_tool_schemas

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a research assistant. Given a topic, use your tools to \
find accurate, current information. Be thorough but concise.

Rules:
- Always search before answering — never rely on prior knowledge alone
- Use web_search for current events, news, and public information
- Use rag_search for internal documents and domain knowledge
- When you have enough information, stop searching and summarise your findings
- Always cite your sources (URLs from web_search, doc_id from rag_search)
"""

# claude-haiku-4-5 pricing
COST_PER_INPUT_TOKEN  = 0.000001    # $1.00 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 0.000005    # $5.00 per 1M output tokens


def _convert_tools_for_anthropic(tool_schemas: list[dict]) -> list[dict]:
    """
    Convert OpenAI-format tool schemas to Anthropic format.

    OpenAI format:  {"type": "function", "function": {"name": ..., "description": ...,
                    "parameters": ...}}
    Anthropic format: {"name": ..., "description": ..., "input_schema": ...}
    """
    anthropic_tools = []
    for tool in tool_schemas:
        fn = tool["function"]
        anthropic_tools.append(
            {
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            }
        )
    return anthropic_tools


def _build_anthropic_messages(messages: list[dict]) -> list[dict]:
    """
    Convert our internal message format to Anthropic's expected format.
    Anthropic does not accept 'tool_calls' in assistant messages —
    it uses 'content' blocks instead.
    Anthropic also does not use the 'name' field in tool result messages.
    """
    anthropic_messages = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            # System messages are passed separately to the API — skip here
            continue

        elif role == "user":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": msg["content"],
                }
            )

        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Anthropic represents tool use as content blocks
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": msg["content"],
                        }
                    )
                for tc in tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    )
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
            else:
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": msg["content"],
                    }
                )

        elif role == "tool":
            # Anthropic expects tool results as user messages with content blocks
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"],
                        }
                    ],
                }
            )

    return anthropic_messages


async def research_node(state: ResearchState) -> dict:
    """
    ReAct agent node using Anthropic SDK directly.
    Runs Thought → Action → Observation loop until FINISH or budget exceeded.
    """
    run_id = state.get("run_id", "unknown")
    topic = state.get("topic", "")
    log.info("research_node_started", run_id=run_id, topic=topic[:80])

    messages = list(state.get("messages", []))
    if not messages:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research this topic thoroughly: {topic}"},
        ]

    # Convert OpenAI-format tool schemas to Anthropic format once
    anthropic_tools = _convert_tools_for_anthropic(get_tool_schemas())

    # Extract model name — strip provider prefix if present
    # e.g. "anthropic/claude-haiku-20240307" → "claude-haiku-20240307"
    model_name = settings.worker_model.replace("anthropic/", "")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    step_count = state.get("step_count", 0)
    tokens_used = state.get("tokens_used", 0)
    cost_usd = state.get("cost_usd", 0.0)

    # ── ReAct loop ─────────────────────────────────────────────────────────────
    while True:
        current_state = {
            **state,
            "step_count": step_count,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
        }
        try:
            check_budget(current_state)
        except BudgetExceeded as e:
            log.warning("budget_exceeded_in_loop", run_id=run_id, reason=e.reason, step=step_count)
            return {
                "messages": messages,
                "step_count": step_count,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "errors": [f"Budget exceeded: {e.reason}"],
            }

        log.info("llm_call_started", run_id=run_id, step=step_count, messages=len(messages))

        # Build Anthropic-format messages (exclude system message)
        anthropic_messages = _build_anthropic_messages(messages)

        # ── Anthropic SDK call ─────────────────────────────────────────────────
        response = await client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=anthropic_messages,
            tools=anthropic_tools,
        )

        # Track usage and cost
        tokens_used += response.usage.input_tokens + response.usage.output_tokens
        cost_usd += (
            response.usage.input_tokens * COST_PER_INPUT_TOKEN
            + response.usage.output_tokens * COST_PER_OUTPUT_TOKEN
        )
        step_count += 1

        # ── Parse response content blocks ─────────────────────────────────────
        # Anthropic returns a list of content blocks — text and/or tool_use
        text_content = ""
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # Append assistant turn to our internal message format
        if tool_use_blocks:
            messages.append(
                {
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": [
                        {
                            "id": tb.id,
                            "type": "function",
                            "function": {
                                "name": tb.name,
                                "arguments": json.dumps(tb.input),
                            },
                        }
                        for tb in tool_use_blocks
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": [],
                }
            )

        # ── FINISH check ───────────────────────────────────────────────────────
        # stop_reason == "end_turn" means the model is done — no tool calls
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            log.info(
                "research_node_finished", run_id=run_id, steps=step_count, cost=round(cost_usd, 6)
            )
            break

        # ── Dispatch tool calls ────────────────────────────────────────────────
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_input = tool_block.input  # already a dict from Anthropic

            log.info("tool_call_dispatched", run_id=run_id, tool=tool_name, step=step_count)

            tool_result = await dispatch(tool_name, tool_input)

            # Store in our internal format — _build_anthropic_messages converts it
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_block.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result),
                }
            )

    return {
        "messages": messages,
        "step_count": step_count,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
    }
