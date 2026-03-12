from langchain_core.documents import Document

from app.cache.redis_cache import RedisCache
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        cache: RedisCache,
        top_k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._cache = cache
        self._top_k = top_k
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    def retrieve(self, query: str) -> list[Document]:
        cache_key = RedisCache.build_key("query", query)
        cached = self._cache.get_json(cache_key)
        if cached is not None:
            return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in cached]

        vector_results = self._vector_retriever.retrieve(query)
        keyword_results = self._keyword_retriever.retrieve(query)

        merged: dict[str, tuple[Document, float]] = {}
        for document, score in vector_results:
            key = str(document.metadata.get("chunk_id", hash(document.page_content)))
            merged[key] = (document, max(score, 0.0) * self._vector_weight)

        for document, score in keyword_results:
            key = str(document.metadata.get("chunk_id", hash(document.page_content)))
            weighted_score = score * self._keyword_weight
            if key in merged:
                existing_document, existing_score = merged[key]
                merged[key] = (existing_document, existing_score + weighted_score)
            else:
                merged[key] = (document, weighted_score)

        ranked_documents = [item[0] for item in sorted(merged.values(), key=lambda value: value[1], reverse=True)]
        top_documents = ranked_documents[: self._top_k]
        self._cache.set_json(
            cache_key,
            [{"page_content": document.page_content, "metadata": document.metadata} for document in top_documents],
        )
        return top_documents