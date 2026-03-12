from langchain_core.documents import Document

from app.vectorstore.vector_repository import VectorRepository


class VectorRetriever:
    def __init__(self, vector_repository: VectorRepository, top_k: int) -> None:
        self._vector_repository = vector_repository
        self._top_k = top_k

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        return self._vector_repository.similarity_search_with_scores(query, self._top_k)