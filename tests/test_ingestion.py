# Tests for the ingestion pipeline.
# All external API calls are mocked — no real API calls, no cost.
#
# Run: uv run pytest tests/test_ingestion.py -v

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helper: build a fake OpenAI embedding response ────────────────────────────
# The real OpenAI response has this structure. We replicate it so our code
# doesn't know the difference between a real call and a mocked one.
def make_fake_embedding_response(n: int, dimensions: int = 1536):
    """Create a fake OpenAI embedding response for n texts."""
    embedding_data = []
    for i in range(n):
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * dimensions  # flat vector — fine for testing
        embedding_data.append(mock_embedding)

    mock_usage = MagicMock()
    mock_usage.total_tokens = n * 100  # fake: 100 tokens per chunk

    mock_response = MagicMock()
    mock_response.data = embedding_data
    mock_response.usage = mock_usage
    return mock_response


# ── Tests ──────────────────────────────────────────────────────────────────────
class TestIngestionPipeline:
    @pytest.mark.asyncio
    @patch("app.ingestion.pipeline.AsyncOpenAI")
    @patch("app.ingestion.pipeline.Pinecone")
    async def test_ingest_short_document(
        self,
        mock_pinecone_cls,
        mock_openai_cls,
    ):
        """
        A short document (under 512 tokens) should produce 1 chunk and 1 vector.
        """
        # Set up mock OpenAI client
        mock_openai = AsyncMock()
        mock_openai_cls.return_value = mock_openai
        mock_openai.embeddings.create = AsyncMock(
            return_value=make_fake_embedding_response(1)  # 1 chunk → 1 embedding
        )

        # Set up mock Pinecone
        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        # Run ingestion
        from app.ingestion.pipeline import ingest_document

        result = await ingest_document(
            text="This is a short test document about LangGraph.",
            doc_id="test-doc-001",
            source="test",
            namespace="test",
        )

        # Assertions
        assert result.doc_id == "test-doc-001"
        assert result.chunk_count >= 1
        assert result.vectors_upserted >= 1
        assert result.tokens_used > 0
        assert result.cost_usd >= 0.0

        # Verify Pinecone upsert was called
        mock_index.upsert.assert_called()

    @pytest.mark.asyncio
    @patch("app.ingestion.pipeline.AsyncOpenAI")
    @patch("app.ingestion.pipeline.Pinecone")
    async def test_ingest_empty_document(
        self,
        mock_pinecone_cls,
        mock_openai_cls,
    ):
        """An empty document should return zero chunks and not call the APIs."""
        mock_openai = AsyncMock()
        mock_openai_cls.return_value = mock_openai

        from app.ingestion.pipeline import ingest_document

        result = await ingest_document(
            text="   ",  # whitespace only
            doc_id="empty-doc",
            source="test",
        )

        assert result.chunk_count == 0
        assert result.vectors_upserted == 0
        # OpenAI should NOT have been called for an empty document
        mock_openai.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.ingestion.pipeline.AsyncOpenAI")
    @patch("app.ingestion.pipeline.Pinecone")
    async def test_idempotency_same_doc_same_ids(
        self,
        mock_pinecone_cls,
        mock_openai_cls,
    ):
        """
        Ingesting the same document twice should produce the same vector IDs.
        This verifies idempotency — no duplicate vectors on re-ingestion.
        """

        mock_openai = AsyncMock()
        mock_openai_cls.return_value = mock_openai

        upserted_ids = []
        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        # Capture what IDs were upserted
        def capture_upsert(vectors, namespace):
            upserted_ids.extend([v["id"] for v in vectors])

        mock_index.upsert.side_effect = capture_upsert

        doc_text = "Test document for idempotency check."
        mock_openai.embeddings.create = AsyncMock(return_value=make_fake_embedding_response(1))

        from app.ingestion.pipeline import ingest_document

        # Ingest once
        await ingest_document(text=doc_text, doc_id="idempotent-doc", source="test")
        ids_first_run = list(upserted_ids)

        # Ingest again — same doc_id
        upserted_ids.clear()
        mock_openai.embeddings.create = AsyncMock(return_value=make_fake_embedding_response(1))
        await ingest_document(text=doc_text, doc_id="idempotent-doc", source="test")
        ids_second_run = list(upserted_ids)

        # IDs must be identical — second run overwrites, doesn't add new vectors
        assert ids_first_run == ids_second_run, (
            "Same document ingested twice produced different vector IDs — "
            "this means duplicate vectors would be created in Pinecone"
        )

    def test_splitter_produces_overlap(self):
        """
        Verify chunking overlap is working — the end of chunk N
        should appear in the beginning of chunk N+1.
        """
        from app.ingestion.pipeline import SPLITTER

        # Create a document long enough to produce multiple chunks
        long_text = " ".join(
            [f"Sentence number {i} about LangGraph and RAG systems." for i in range(200)]
        )
        chunks = SPLITTER.split_text(long_text)

        assert len(chunks) >= 2, "Document should produce multiple chunks"

        # The last ~200 chars of chunk 0 should appear somewhere in chunk 1
        # (this is what overlap does)
        end_of_chunk_0 = chunks[0][-100:]  # last 100 chars
        assert end_of_chunk_0 in chunks[1], (
            "Overlap not working — end of chunk 0 should appear at start of chunk 1"
        )
