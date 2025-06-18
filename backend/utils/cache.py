import json
import pickle
import logging

import redis.asyncio as redis

from datetime import timedelta
from typing import Any, Optional
from core.config import settings

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)

async def cache_result(
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialization: str = "json"
) -> bool:
    """
    Cache a result with optional TTL

    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
        serialization: "json" or "pickle"
    """
    try:
        if serialization == "json":
            serialized = json.dumps(value)
        else:
            serialized = pickle.dumps(value)

        if ttl:
            await redis_client.setex(key, ttl, serialized)
        else:
            await redis_client.set(key, serialized)

        return True

    except Exception as e:
        logger.error(f"Cache set failed for key {key}: {e}")
        return False

async def get_cached_result(
        key: str,
        serialization: str = "json"
) -> Optional[Any]:
    """
    Get cached result

    Args:
        key: Cache key
        serialization: "json" or "pickle"
    """
    try:
        cached = await redis_client.get(key)

        if cached is None:
            return None

        if serialization == "json":
            return json.loads(cached)
        else:
            return pickle.loads(cached)

    except Exception as e:
        logger.error(f"Cache get failed for key {key}: {e}")
        return None

async def delete_cached_result(key: str) -> bool:
    """Delete cached result"""
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Cache delete failed for key {key}: {e}")
        return False

async def invalidate_pattern(pattern: str) -> int:
    """
    Invalidate all keys matching a pattern

    Args:
        pattern: Redis pattern (e.g., "match_pair:*")

    Returns:
        Number of keys deleted
    """
    try:
        keys = []
        async for key in redis_client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            return await redis_client.delete(*keys)
        return 0

    except Exception as e:
        logger.error(f"Pattern invalidation failed for {pattern}: {e}")
        return 0

async def get_cache_stats() -> dict:
    """Get cache statistics"""
    try:
        info = await redis_client.info()
        return {
            "used_memory": info.get("used_memory_human", "0"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
            ) if info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {}

def cached(ttl: int = 3600, key_prefix: str = "func"):
    """
    Decorator to cache function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{hash((args, tuple(kwargs.items())))}"

            result = await get_cached_result(cache_key)
            if result is not None:
                return result

            result = await func(*args, **kwargs)

            await cache_result(cache_key, result, ttl)

            return result

        return wrapper
    return decorator