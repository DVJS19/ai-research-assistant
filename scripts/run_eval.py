#
# Runs the golden set against the live agent and reports recall scores.
# Server must be running: uv run uvicorn app.main:app --reload
#
# Usage:
#   uv run python scripts/run_eval.py
#   uv run python scripts/run_eval.py --url http://localhost:8000

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_SET_PATH = Path("evals/golden_set.json")
RESULTS_DIR = Path("evals/results")


async def run_single_eval(
    client: httpx.AsyncClient,
    question: dict,
    base_url: str,
) -> dict:
    """Run one golden set question and score the result."""
    topic = question["topic"]
    expected = question["expected_topics"]
    q_id = question["id"]

    try:
        response = await client.post(
            f"{base_url}/research",
            json={"topic": topic},
            timeout=120.0,  # research can take up to 2 minutes
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {
            "id": q_id,
            "topic": topic,
            "status": "error",
            "error": str(e),
            "recall": 0.0,
            "topics_found": [],
            "topics_missed": expected,
        }

    answer = result.get("answer", "").lower()

    # Check which expected topics appear in the answer
    topics_found = [t for t in expected if t.lower() in answer]
    topics_missed = [t for t in expected if t.lower() not in answer]
    recall = len(topics_found) / len(expected) if expected else 0.0

    return {
        "id": q_id,
        "topic": topic,
        "status": "completed",
        "recall": round(recall, 3),
        "topics_found": topics_found,
        "topics_missed": topics_missed,
        "steps_taken": result.get("steps_taken", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "trace_id": result.get("trace_id", ""),
    }


async def main(base_url: str) -> None:
    if not GOLDEN_SET_PATH.exists():
        print(f"Golden set not found: {GOLDEN_SET_PATH}")
        sys.exit(1)

    questions = json.loads(GOLDEN_SET_PATH.read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running eval against {base_url}")
    print(f"Questions: {len(questions)}")
    print()

    results = []
    total_cost = 0.0

    # Run questions sequentially — avoid hammering the API
    async with httpx.AsyncClient() as client:
        for i, question in enumerate(questions):
            print(f"[{i + 1}/{len(questions)}] {question['topic'][:60]}...")
            result = await run_single_eval(client, question, base_url)
            results.append(result)

            recall_pct = result["recall"] * 100
            cost = result.get("cost_usd", 0.0)
            total_cost += cost

            status_icon = "✓" if result["recall"] >= 0.6 else "✗"
            print(
                f"    {status_icon} recall: {recall_pct:.0f}%  "
                f"steps: {result.get('steps_taken', '-')}  "
                f"cost: ${cost:.5f}"
            )

            if result["topics_missed"]:
                print(f"    missed: {result['topics_missed']}")
            print()

    # Summary
    successful = [r for r in results if r["status"] == "completed"]
    avg_recall = sum(r["recall"] for r in successful) / len(successful) if successful else 0
    pass_count = sum(1 for r in successful if r["recall"] >= 0.6)

    print("─" * 60)
    print(f"Results:     {pass_count}/{len(questions)} passed (recall ≥ 60%)")
    print(f"Avg recall:  {avg_recall * 100:.1f}%")
    print(f"Total cost:  ${total_cost:.4f}")
    print(f"Avg cost:    ${total_cost / len(questions):.5f} per question")
    print()

    # Save results
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": len(questions),
                    "passed": pass_count,
                    "avg_recall": round(avg_recall, 3),
                    "total_cost": round(total_cost, 6),
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run golden set evaluation")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL of the research API"
    )
    args = parser.parse_args()
    asyncio.run(main(args.url))
