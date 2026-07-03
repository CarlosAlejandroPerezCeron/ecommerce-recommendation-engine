"""Redis client wrapper with TTL caching, candidate pool pre-warming, and feedback stream."""
import json, logging
from typing import Any, Optional
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)
DEFAULT_TTL = 300
CANDIDATE_POOL_TTL = 600

class RedisClient:
    def __init__(self, host="localhost", port=6379, db=0):
        self._pool = aioredis.ConnectionPool.from_url(
            f"redis://{host}:{port}/{db}", max_connections=50, decode_responses=True)
        self._client = aioredis.Redis(connection_pool=self._pool)

    async def get(self, key: str) -> Optional[Any]:
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Redis GET %s: %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
            return True
        except Exception as e:
            logger.warning("Redis SET %s: %s", key, e)
            return False

    async def delete(self, pattern: str) -> int:
        try:
            keys = [k async for k in self._client.scan_iter(match=pattern)]
            return await self._client.delete(*keys) if keys else 0
        except Exception as e:
            logger.warning("Redis DELETE %s: %s", pattern, e)
            return 0

    async def xadd(self, stream: str, data: dict) -> Optional[str]:
        try:
            return await self._client.xadd(stream, data, maxlen=100_000, approximate=True)
        except Exception as e:
            logger.warning("Redis XADD %s: %s", stream, e)
            return None

    async def get_candidate_pool(self, context: str) -> list:
        pool = await self.get(f"candidate_pool:{context}")
        if pool is None:
            logger.warning("Candidate pool cache miss for context=%s", context)
            return []
        return pool

    async def warm_candidate_pool(self, context: str, skus: list) -> bool:
        return await self.set(f"candidate_pool:{context}", skus, ttl=CANDIDATE_POOL_TTL)

    async def ping(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
