from app.config import settings
from app.logger import get_logger
from app.tools.rag_search import rag_search

log = get_logger(__name__)


async def kb_research_node(state: dict) -> dict:
    """
    Dedicated knowledge base research worker.

    Searches Pinecone directly — no LLM reasoning needed for retrieval.
    The synthesis node decides how to use these results.
    """
    topic = state.get("topic", "")
    run_id = state.get("run_id", "unknown")

    log.info("kb_research_started", run_id=run_id, topic=topic[:60])

    try:
        # Run two searches with slightly different queries
        # to maximise recall from the knowledge base
        result_1 = await rag_search(
            query=topic,
            top_k=5,
            namespace=settings.pinecone_namespace,
        )
        result_2 = await rag_search(
            query=f"{topic} implementation details examples",
            top_k=3,
            namespace=settings.pinecone_namespace,
        )

        # Merge results, deduplicate by doc_id + chunk_index
        seen = set()
        combined = []
        for r in result_1["results"] + result_2["results"]:
            key = f"{r['doc_id']}::{r['chunk_index']}"
            if key not in seen:
                seen.add(key)
                combined.append(r)

        # Sort by score descending
        combined.sort(key=lambda x: x["score"], reverse=True)

        confidence = combined[0]["score"] if combined else 0.0

        log.info(
            "kb_research_completed",
            run_id=run_id,
            chunks_found=len(combined),
            confidence=confidence,
        )

        return {
            "kb_result": {
                "chunks": combined[:8],  # top 8 after merge
                "confidence": confidence,
                "source": "pinecone",
            }
        }

    except Exception as e:
        log.error("kb_research_failed", run_id=run_id, error=str(e))
        return {
            "kb_result": {
                "chunks": [],
                "confidence": 0.0,
                "error": str(e),
            }
        }
