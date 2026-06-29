import httpx

from app.logger import get_logger

log = get_logger(__name__)

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"


async def wiki_research_node(state: dict) -> dict:
    """
    Dedicated Wikipedia research worker.

    Fetches structured encyclopaedia content for the topic.
    No LLM needed — Wikipedia REST API returns clean plain text.
    No API key needed — Wikipedia is public.
    """
    topic = state.get("topic", "")
    run_id = state.get("run_id", "unknown")

    log.info("wiki_research_started", run_id=run_id, topic=topic[:60])

    async with httpx.AsyncClient(
        timeout=10.0,
        headers={"User-Agent": "AIResearchAssistant/1.0 (portfolio project; contact@example.com)"}
        ) as client:
        # Step 1: Search Wikipedia for the best matching article
         # Truncate to first 50 chars and strip question marks
         # Wikipedia opensearch works better with short noun phrases
        wiki_query = topic[:50].split("?")[0].strip()
        search_response = await client.get(
            WIKI_SEARCH_API,
            params={
                "action": "opensearch",
                "search": wiki_query,
                "limit": 3,
                "format": "json",
            },
        )
       
        try:
            search_data = search_response.json()
        except Exception:
            log.warning("wiki_search_parse_failed", topic=topic[:60])
            return {
                "wiki_result": {
                    "extract": "",
                    "title": "",
                    "url": "",
                    "confidence": 0.0,
                }
            }
        titles = search_data[1] if len(search_data) > 1 else []

        if not titles:
            log.info("wiki_no_results", run_id=run_id, topic=topic[:60])
            return {
                "wiki_result": {
                    "extract": "",
                    "title": "",
                    "url": "",
                    "confidence": 0.0,
                }
            }

        # Step 2: Fetch the summary for the top result
        best_title = titles[0]
        encoded_title = best_title.replace(" ", "_")

        try:
            summary_response = await client.get(WIKI_API.format(encoded_title))
            summary_response.raise_for_status()
            data = summary_response.json()

            extract = data.get("extract", "")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            confidence = 0.7 if extract else 0.0

            log.info(
                "wiki_research_completed",
                run_id=run_id,
                title=best_title,
                extract_len=len(extract),
                confidence=confidence,
            )

            return {
                "wiki_result": {
                    "extract": extract,
                    "title": best_title,
                    "url": page_url,
                    "confidence": confidence,
                }
            }

        except Exception as e:
            log.warning("wiki_fetch_failed", run_id=run_id, title=best_title, error=str(e))
            return {
                "wiki_result": {
                    "extract": "",
                    "title": best_title,
                    "url": "",
                    "confidence": 0.0,
                    "error": str(e),
                }
            }
