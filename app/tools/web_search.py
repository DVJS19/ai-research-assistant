import httpx

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Maximum results to return to the agent.
# More results = more context = higher token cost.
# 5 is the sweet spot for research tasks.
MAX_RESULTS = 5


async def web_search(query: str, max_results: int = MAX_RESULTS) -> dict:
    """
    Search the web using the Brave Search API.

    Called by the tool dispatcher when the agent issues a web_search tool call.

    Args:
        query:       The search query string.
        max_results: Number of results to return. Capped at MAX_RESULTS.

    Returns:
        {
            results: list of {title, url, snippet},
            query:   original query string,
            count:   number of results returned,
        }
    """
    max_results = min(max_results, MAX_RESULTS)

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.brave_search_api_key,
    }
    params = {
        "q": query,
        "count": max_results,
        # freshness=None means any age — set to "pd" for past day if needed
    }

    log.info("web_search_started", query=query[:80], max_results=max_results)

    # httpx.AsyncClient keeps the connection async — never blocks the event loop.
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

    # Extract only what the agent needs — title, url, snippet.
    # Brave returns a lot more but we discard it to save context window tokens.
    web_results = data.get("web", {}).get("results", [])
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        }
        for r in web_results[:max_results]
    ]

    log.info("web_search_completed", query=query[:80], count=len(results))

    return {
        "results": results,
        "query": query,
        "count": len(results),
    }
