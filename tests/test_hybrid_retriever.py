from langchain_core.documents import Document

from app.cache.redis_cache import RedisCache
from app.retrieval.hybrid_retriever import HybridRetriever


class StubVectorRetriever:
    def retrieve(self, query: str):
        return [
            (
                Document(
                    page_content="Refunds require manager approval.",
                    metadata={"chunk_id": "chunk-1", "document_name": "policy.txt"},
                ),
                0.9,
            )
        ]


class StubKeywordRetriever:
    def retrieve(self, query: str):
        return [
            (
                Document(
                    page_content="Refunds require manager approval.",
                    metadata={"chunk_id": "chunk-1", "document_name": "policy.txt"},
                ),
                0.7,
            ),
            (
                Document(
                    page_content="Vacation requests need 2 weeks notice.",
                    metadata={"chunk_id": "chunk-2", "document_name": "hr.txt"},
                ),
                0.6,
            ),
        ]


def test_hybrid_retriever_merges_vector_and_keyword_results() -> None:
    cache = RedisCache("redis://localhost:1/0", 60)
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(),
        keyword_retriever=StubKeywordRetriever(),
        cache=cache,
        top_k=2,
        vector_weight=0.7,
        keyword_weight=0.3,
    )

    documents = retriever.retrieve("refund approval")

    assert len(documents) == 2
    assert documents[0].metadata["chunk_id"] == "chunk-1"