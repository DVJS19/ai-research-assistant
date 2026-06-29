import json

import anthropic

from app.config import settings
from app.logger import get_logger
from app.tools.dispatcher import dispatch

log = get_logger(__name__)

# Per-worker budget — fraction of total run budget
MAX_STEPS = 4
MAX_COST_USD = 0.50

WEB_SYSTEM_PROMPT = """You are a web research specialist. Your only job is to \
search the web for current, accurate information about the given topic.

Rules:
- Use web_search to find relevant information
- Search 2-3 times with different queries to get comprehensive coverage
- Focus on recent and authoritative sources
- Return a concise structured summary of your findings
- Always include source URLs
"""


async def web_research_node(state: dict) -> dict:
    """
    Dedicated web research worker.

    Runs its own focused ReAct loop using only web_search.
    Returns web_result for the synthesis node to read.
    """
    topic = state.get("topic", "")
    run_id = state.get("run_id", "unknown")

    log.info("web_research_started", run_id=run_id, topic=topic[:60])

    # Only give this worker the web_search tool
    web_tool_schema = [
        {
            "name": "web_search",
            "description": "Search the web for current information about a topic.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        }
    ]

    messages = [{"role": "user", "content": f"Research this topic using web search: {topic}"}]

    model_name = settings.worker_model.replace("anthropic/", "")
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    step_count = 0
    cost_usd = 0.0
    sources = []
    findings = ""

    while step_count < MAX_STEPS and cost_usd < MAX_COST_USD:
        response = await client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=WEB_SYSTEM_PROMPT,
            messages=messages,
            tools=web_tool_schema,
        )

        cost_usd += (
            response.usage.input_tokens * 0.00000025 + response.usage.output_tokens * 0.00000125
        )
        step_count += 1

        # Parse content blocks
        text_content = ""
        tool_use_blocks = []
        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # Append assistant turn
        if tool_use_blocks:
            content_blocks = []
    for block in response.content:
        if block.type == "text":
            content_blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": content_blocks,
                }
            )
        else:
            findings = text_content
            messages.append({"role": "assistant", "content": text_content})

        # FINISH — no tool calls means agent is done
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        # Dispatch tool calls and append results
        for tool_block in tool_use_blocks:
            tool_result = await dispatch(tool_block.name, tool_block.input)

            # Collect sources from web results
            for r in tool_result.get("results", []):
                if r.get("url"):
                    sources.append({"url": r["url"], "title": r.get("title", "")})

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": json.dumps(tool_result),
                        }
                    ],
                }
            )

    # Score based on how much we found
    if sources and findings and len(sources) >= 5:
        confidence = 0.95
    elif sources and len(sources) >= 5:
        confidence = 0.80   # ← has sources but no findings text
    elif sources and findings:
        confidence = 0.85
    elif sources or findings:
        confidence = 0.5
    else:
        confidence = 0.1
    
    # ── End of while loop ─────────────────────────────────────────────────────

    # If agent hit step limit without writing findings,
    # use the last assistant text message as findings
    if not findings:
        for msg in reversed(messages):
            content = msg.get("content", "")
            if msg.get("role") == "assistant" and isinstance(content, str) and content:
                findings = content
                break

    return {
        "web_result": {
            "findings": findings,
            "sources": sources,
            "confidence": confidence,
            "steps": step_count,
            "cost_usd": cost_usd,
        }
    }
