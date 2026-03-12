from hashlib import sha256

from langchain_openai import OpenAIEmbeddings

from app.cache.redis_cache import RedisCache
from app.core.config import Settings


class CachedOpenAIEmbeddings:
    def __init__(self, client: OpenAIEmbeddings, cache: RedisCache) -> None:
        self._client = client
        self._cache = cache

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            key = self._build_key("embedding:document", text)
            cached_value = self._cache.get_json(key)
            if cached_value is None:
                vectors.append(None)
                missing_indices.append(index)
                missing_texts.append(text)
            else:
                vectors.append(cached_value)

        if missing_texts:
            generated = self._client.embed_documents(missing_texts)
            for index, text, vector in zip(missing_indices, missing_texts, generated, strict=True):
                vectors[index] = vector
                self._cache.set_json(self._build_key("embedding:document", text), vector)

        return [vector for vector in vectors if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        key = self._build_key("embedding:query", text)
        cached_value = self._cache.get_json(key)
        if cached_value is not None:
            return cached_value

        embedding = self._client.embed_query(text)
        self._cache.set_json(key, embedding)
        return embedding

    @staticmethod
    def _build_key(namespace: str, text: str) -> str:
        return f"{namespace}:{sha256(text.encode('utf-8')).hexdigest()}"


class EmbeddingService:
    def __init__(self, settings: Settings, cache: RedisCache) -> None:
        embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
            timeout=settings.openai_request_timeout,
        )
        self._embeddings = CachedOpenAIEmbeddings(embeddings, cache)

    def get_embedding_model(self) -> CachedOpenAIEmbeddings:
        return self._embeddings