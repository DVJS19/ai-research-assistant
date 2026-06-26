import cohere
from openai import AsyncOpenAI
from pinecone import Pinecone

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

# Minimum reranker score to include a chunk in results.
# Below this the KB probably doesn't have relevant info for this query.
RELEVANCE_THRESHOLD = 0.4


async def rag_search(
    query: str,
    top_k: int = 5,
    namespace: str = "default",
) -> dict:
    """
    Hybrid RAG search: embed query → Pinecone ANN → Cohere rerank.

    This is the function called by the tool registry handler.
    The agent calls this via the 'rag_search' tool.

    Returns a dict with:
        results:    list of {text, doc_id, score, chunk_index}
        confidence: float 0-1 based on top result score
        query:      the original query (for logging)
    """
    log.info("rag_search_started", query=query[:80], top_k=top_k, namespace=namespace)

    # ── Step 1: Embed the query ───────────────────────────────────────────────
    # MUST use the same model as ingestion — text-embedding-3-small.
    # Different model = different vector space = meaningless similarity scores.
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    embed_response = await openai_client.embeddings.create(
        model=settings.embedding_model,  # text-embedding-3-small
        input=query,
    )
    query_vector = embed_response.data[0].embedding

    # ── Step 2: ANN search in Pinecone (retrieve top 20 candidates) ───────────
    # We retrieve 20 initially — more than we need — to give the reranker
    # enough candidates to pick the best 5 from.
    # More candidates = better reranker precision, slightly more latency.
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    search_response = index.query(
        vector=query_vector,
        top_k=20,
        namespace=namespace,
        include_metadata=True,  # we need the text field from metadata
    )

    candidates = search_response.matches
    if not candidates:
        log.info("rag_search_no_results", query=query[:80])
        return {"results": [], "confidence": 0.0, "query": query}

    # ── Step 3: Rerank with Cohere ────────────────────────────────────────────
    # Cohere's cross-encoder reads (query, document) pairs together.
    # Much higher precision than cosine similarity alone.
    co = cohere.AsyncClient(api_key=settings.cohere_api_key)

    # Extract text from Pinecone metadata for reranking
    docs_to_rerank = [match.metadata.get("text", "") for match in candidates]

    rerank_response = await co.rerank(
        query=query,
        documents=docs_to_rerank,
        top_n=top_k,  # return top_k after reranking (default 5)
        model="rerank-english-v3.0",
    )

    # ── Step 4: Filter by relevance threshold and build result ────────────────
    results = []
    for item in rerank_response.results:
        if item.relevance_score < RELEVANCE_THRESHOLD:
            continue  # skip low-confidence results

        original_match = candidates[item.index]
        results.append(
            {
                "text": original_match.metadata.get("text", ""),
                "doc_id": original_match.metadata.get("doc_id", ""),
                "source": original_match.metadata.get("source", ""),
                "chunk_index": original_match.metadata.get("chunk_index", 0),
                "score": round(item.relevance_score, 4),
            }
        )

    # Confidence = top result's reranker score (or 0 if no results passed threshold)
    confidence = results[0]["score"] if results else 0.0

    log.info(
        "rag_search_completed",
        query=query[:80],
        candidates=len(candidates),
        results_returned=len(results),
        confidence=confidence,
    )

    return {
        "results": results,
        "confidence": confidence,
        "query": query,
    }
