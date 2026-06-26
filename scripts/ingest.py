#
# Usage:
#   uv run python scripts/ingest.py --source data/docs/ --namespace default
#   uv run python scripts/ingest.py --source data/docs/article.txt --namespace research
#
# This script reads text files from a directory (or a single file),
# runs them through the ingestion pipeline, and upserts to Pinecone.

import argparse
import asyncio

# Add parent to path so we can import from app/
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import ingest_document
from app.logger import get_logger, setup_logging

log = get_logger(__name__)


async def ingest_file(path: Path, namespace: str) -> None:
    """Ingest a single text file."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        log.warning("empty_file_skipped", path=str(path))
        return

    # Use the file path as the doc_id — ensures idempotency on re-ingest
    doc_id = str(path)
    source = path.suffix.lstrip(".") or "txt"  # "pdf", "txt", "md" etc.

    result = await ingest_document(
        text=text,
        doc_id=doc_id,
        source=source,
        namespace=namespace,
        metadata={"filename": path.name},
    )

    print(
        f"✓ {path.name}: {result.chunk_count} chunks, "
        f"{result.vectors_upserted} vectors, "
        f"${result.cost_usd:.5f}"
    )


async def main(source: str, namespace: str) -> None:
    setup_logging()
    source_path = Path(source)

    if source_path.is_file():
        # Single file
        files = [source_path]
    elif source_path.is_dir():
        # All .txt, .md, .pdf files in the directory (not recursive)
        files = (
            list(source_path.glob("*.txt"))
            + list(source_path.glob("*.md"))
            + list(source_path.glob("*.pdf"))
        )
        files.sort()
    else:
        print(f"Error: {source} is not a file or directory")
        sys.exit(1)

    if not files:
        print(f"No .txt / .md / .pdf files found in {source}")
        sys.exit(1)

    print(f"Ingesting {len(files)} file(s) into namespace '{namespace}'...")
    print()

    total_chunks = 0
    total_vectors = 0
    total_cost_usd = 0.0

    for f in files:
        try:
            # We run files one at a time here to keep it simple and readable.
            # For production you would use asyncio.gather() to parallelise.
            result = await ingest_file(f, namespace)
            if result:
                total_chunks += result.chunk_count
                total_vectors += result.vectors_upserted
                total_cost_usd += result.cost_usd
        except Exception as e:
            print(f"✗ {f.name}: {e}")

    print()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument("--source", required=True, help="File or directory to ingest")
    parser.add_argument("--namespace", default="default", help="Pinecone namespace")
    args = parser.parse_args()

    asyncio.run(main(args.source, args.namespace))
