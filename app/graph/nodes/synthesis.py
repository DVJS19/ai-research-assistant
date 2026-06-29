import anthropic

from app.config import settings
from app.graph.state import ResearchState
from app.logger import get_logger

log = get_logger(__name__)

SYNTHESIS_PROMPT = """You are a research synthesis specialist. You have been given \
research findings from three independent sources: web search, an internal knowledge \
base, and Wikipedia. Your job is to synthesise these into a comprehensive, accurate \
research report.

Instructions:
- Cross-reference findings across sources — highlight agreements and contradictions
- Prioritise the internal knowledge base for technical specifics
- Use web results for current information and recent developments
- Use Wikipedia for background context and definitions
- Structure the report clearly with sections
- Cite your sources throughout
- Be concise but comprehensive
"""


def _calculate_confidence(
    web_result: dict | None,
    kb_result: dict | None,
    wiki_result: dict | None,
) -> tuple[dict[str, float], float]:
    """
    Calculate per-source confidence and overall report confidence.

    Returns (confidence_scores, overall_confidence).
    """
    scores = {
        "web": web_result.get("confidence", 0.0) if web_result else 0.0,
        "kb": kb_result.get("confidence", 0.0) if kb_result else 0.0,
        "wiki": wiki_result.get("confidence", 0.0) if wiki_result else 0.0,
    }

    # Weighted average — KB and web weighted higher than wiki
    weights = {"web": 0.45, "kb": 0.45}
    weighted = sum(scores[k] * weights[k] for k in ["web","kb"])

    # Only penalise if BOTH primary sources (web + kb) are missing
    # Wiki failure alone should not tank confidence
    primary_missing = sum(1 for k in ["web", "kb"] if scores[k] == 0.0)
    penalty = primary_missing * 0.20

    overall = max(weighted - penalty, 0.0)
    return scores, round(overall, 3)


async def synthesis_node(state: ResearchState) -> dict:
    """
    Synthesis agent — cross-references all worker results and produces
    the final structured report with confidence scoring.

    Uses claude-sonnet-4-6 — higher quality justified for final output.
    """
    run_id = state.get("run_id", "unknown")
    topic = state.get("topic", "")
    web_result = state.get("web_result")
    kb_result = state.get("kb_result")
    wiki_result = state.get("wiki_result")

    log.info(
        "synthesis_started",
        run_id=run_id,
        has_web=web_result is not None,
        has_kb=kb_result is not None,
        has_wiki=wiki_result is not None,
    )

    # Calculate confidence before synthesis
    confidence_scores, overall_confidence = _calculate_confidence(
        web_result, kb_result, wiki_result
    )

    # Build context for synthesis — combine all worker outputs
    context_parts = [f"Topic: {topic}\n"]

    if web_result and web_result.get("findings"):
        context_parts.append(f"## Web Research Findings\n{web_result['findings']}\n")

    if kb_result and kb_result.get("chunks"):
        kb_text = "\n\n".join(
            f"[{c['doc_id']} chunk {c['chunk_index']}]\n{c['text']}"
            for c in kb_result["chunks"][:5]
        )
        context_parts.append(f"## Internal Knowledge Base\n{kb_text}\n")

    if wiki_result and wiki_result.get("extract"):
        context_parts.append(f"## Wikipedia Background\n{wiki_result['extract']}\n")

    context = "\n".join(context_parts)

    # Use the synthesis model — larger, higher quality
    model_name = settings.synthesis_model.replace("anthropic/", "")
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=SYNTHESIS_PROMPT,
        messages=[
            {"role": "user", "content": f"Synthesise the following research findings:\n\n{context}"}
        ],
    )

    report_text = response.content[0].text if response.content else ""
    cost_usd = response.usage.input_tokens * 0.000003 + response.usage.output_tokens * 0.000015

    # Collect all sources
    sources = []
    if web_result:
        sources.extend(web_result.get("sources", []))
    if wiki_result and wiki_result.get("url"):
        sources.append(
            {
                "url": wiki_result["url"],
                "title": f"Wikipedia: {wiki_result.get('title', '')}",
            }
        )
    if kb_result:
        for chunk in kb_result.get("chunks", []):
            sources.append(
                {
                    "doc_id": chunk["doc_id"],
                    "score": chunk["score"],
                }
            )

    log.info(
        "synthesis_completed", run_id=run_id, confidence=overall_confidence, cost=round(cost_usd, 6)
    )

    report = {
        "text": report_text,
        "sources": sources,
        "cost_usd": cost_usd,
    }

    return {
        "report": report,
        "confidence_scores": confidence_scores,
        "overall_confidence": overall_confidence,
        "cost_usd": state.get("cost_usd", 0.0) + cost_usd,
    }
