import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pinecone import Pinecone

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)


@dataclass
class IngestResult:
    """Returned after ingesting one document."""

    doc_id: str
    chunk_count: int
    vectors_upserted: int
    tokens_used: int
    cost_usd: float


# ── Chunking config ────────────────────────────────────────────────────────────
# 512 tokens ≈ 2048 characters for typical English text (1 token ≈ 4 chars).
# overlap = 10% = 51 tokens ≈ 200 characters.
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # characters (approx 512 tokens)
    chunk_overlap=200,  # characters (approx 51 tokens, 10%)
    separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
)


async def ingest_document(
    text: str,
    doc_id: str,
    source: str,
    namespace: str = "default",
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
    """
    Ingest one document into Pinecone.

    Args:
        text:      The full document text (already parsed to plain text).
        doc_id:    Unique identifier for this document (e.g. filename or URL).
        source:    Human-readable source label (e.g. "web", "pdf", "wiki").
        namespace: Pinecone namespace — use for multi-tenancy or topic isolation.
        metadata:  Any extra metadata to attach to every chunk from this doc.

    Returns:
        IngestResult with counts and cost.
    """
    # ── Step 1: Chunk ─────────────────────────────────────────────────────────
    chunks = SPLITTER.split_text(text)
    log.info("chunking_completed", doc_id=doc_id, chunk_count=len(chunks))

    if not chunks:
        log.warning("empty_document", doc_id=doc_id)
        return IngestResult(
            doc_id=doc_id, chunk_count=0, vectors_upserted=0, tokens_used=0, cost_usd=0.0
        )

    # ── Step 2: Embed in batches of 64 ───────────────────────────────────────
    # We use the SAME model as the query embedding later.
    # If you change this model, you must re-ingest everything.
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    BATCH_SIZE = 64
    all_embeddings = []
    total_tokens = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        response = await client.embeddings.create(
            model=settings.embedding_model,  # text-embedding-3-small
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])
        total_tokens += response.usage.total_tokens
        log.info(
            "embedding_batch_completed",
            doc_id=doc_id,
            batch=i // BATCH_SIZE,
            tokens=response.usage.total_tokens,
        )

    # ── Step 3: Build vectors with metadata ──────────────────────────────────
    # The vector ID is deterministic: SHA-256(doc_id + chunk_index).
    # Re-ingesting the same document will OVERWRITE existing vectors (idempotent).
    vectors = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, all_embeddings)):
        vector_id = hashlib.sha256(f"{doc_id}::{i}".encode()).hexdigest()
        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk_text,  # stored so we can retrieve readable text
                    **(metadata or {}),  # any extra metadata passed in
                },
            }
        )

    # ── Step 4: Upsert to Pinecone ────────────────────────────────────────────
    # Upsert = insert if new, update if exists.
    # We send in batches of 100 (Pinecone's recommended batch size).
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    UPSERT_BATCH = 100
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=namespace)

    # ── Step 5: Calculate cost and log completion ─────────────────────────────
    # text-embedding-3-small: $0.02 per 1M tokens
    cost_usd = (total_tokens / 1_000_000) * 0.02

    log.info(
        "ingestion_completed",
        doc_id=doc_id,
        chunk_count=len(chunks),
        vectors_upserted=len(vectors),
        tokens_used=total_tokens,
        cost_usd=round(cost_usd, 6),
    )

    return IngestResult(
        doc_id=doc_id,
        chunk_count=len(chunks),
        vectors_upserted=len(vectors),
        tokens_used=total_tokens,
        cost_usd=cost_usd,
    )
