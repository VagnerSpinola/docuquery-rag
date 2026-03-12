import json
import logging
from hashlib import sha256
from typing import Any

from redis import Redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, redis_url: str, default_ttl_seconds: int = 3600) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        self._memory_fallback: dict[str, str] = {}
        try:
            self._client = Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self._enabled = True
        except RedisError:
            logger.warning("Redis unavailable. Falling back to in-memory cache.")
            self._client = None
            self._enabled = False

    @staticmethod
    def build_key(namespace: str, raw_value: str) -> str:
        return f"{namespace}:{sha256(raw_value.encode('utf-8')).hexdigest()}"

    def get_json(self, key: str) -> Any | None:
        raw = self._get(key)
        if raw is None:
            return None

        return json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        self._set(key, json.dumps(value), ttl_seconds)

    def get_text(self, key: str) -> str | None:
        return self._get(key)

    def set_text(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._set(key, value, ttl_seconds)

    def ping(self) -> bool:
        if self._enabled and self._client is not None:
            try:
                return bool(self._client.ping())
            except RedisError:
                return False

        return True

    def _get(self, key: str) -> str | None:
        if self._enabled and self._client is not None:
            try:
                return self._client.get(key)
            except RedisError:
                logger.warning("Redis get failed for key %s.", key)
                return None

        return self._memory_fallback.get(key)

    def _set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self._default_ttl_seconds
        if self._enabled and self._client is not None:
            try:
                self._client.set(name=key, value=value, ex=ttl)
                return
            except RedisError:
                logger.warning("Redis set failed for key %s.", key)

        self._memory_fallback[key] = value