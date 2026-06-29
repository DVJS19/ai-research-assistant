# app/cache.py

import hashlib
import json

import redis.asyncio as aioredis

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

# Cache TTL — 1 hour. Research results can go stale with new web content.
CACHE_TTL_SECONDS = 3600

# Minimum confidence to cache a result.
# Below this the result had missing sections — don't serve it from cache.
MIN_CONFIDENCE_TO_CACHE = 0.75

# Module-level Redis client — initialised once at startup
_redis: aioredis.Redis | None = None


async def setup_cache() -> None:
    """
    Initialise the Redis connection pool.
    Called once at app startup in lifespan.
    """
    global _redis
    _redis = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=2,
    )
    # Verify connection
    await _redis.ping()
    log.info("redis_cache_ready", url=settings.redis_url)


async def close_cache() -> None:
    """Close Redis connection at app shutdown."""
    global _redis
    if _redis:
        await _redis.aclose()
        log.info("redis_cache_closed")


def _cache_key(topic: str) -> str:
    """
    Deterministic cache key from topic string.
    Normalises case and whitespace so minor variations hit the same key.
    """
    normalised = topic.strip().lower()
    return "research:" + hashlib.sha256(normalised.encode()).hexdigest()


async def get_cached_result(topic: str) -> dict | None:
    """
    Check Redis for a cached research result.

    Returns the cached result dict if found, None if miss or Redis unavailable.
    Never raises — cache failures are non-fatal.
    """
    if _redis is None:
        return None

    key = _cache_key(topic)
    try:
        cached = await _redis.get(key)
        if cached:
            result = json.loads(cached)
            log.info("cache_hit", topic=topic[:60], key=key[:16])
            return result
        log.info("cache_miss", topic=topic[:60], key=key[:16])
        return None
    except Exception as e:
        # Redis failure is non-fatal — fall through to full graph execution
        log.warning("cache_read_failed", topic=topic[:60], error=str(e))
        return None


async def set_cached_result(
    topic: str,
    result: dict,
    confidence: float,
) -> None:
    """
    Cache a research result in Redis.

    Only caches if confidence >= MIN_CONFIDENCE_TO_CACHE.
    Low-confidence results (missing workers, partial data) should not be cached
    because the next run might produce a better result.
    """
    if _redis is None:
        return

    if confidence < MIN_CONFIDENCE_TO_CACHE:
        log.info("cache_write_skipped_low_confidence", topic=topic[:60], confidence=confidence)
        return

    key = _cache_key(topic)
    try:
        await _redis.setex(
            name=key,
            time=CACHE_TTL_SECONDS,
            value=json.dumps(result),
        )
        log.info(
            "cache_write_success", topic=topic[:60], confidence=confidence, ttl=CACHE_TTL_SECONDS
        )
    except Exception as e:
        # Cache write failure is non-fatal — result still returned to user
        log.warning("cache_write_failed", topic=topic[:60], error=str(e))
