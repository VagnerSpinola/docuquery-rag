import re

from langchain_core.documents import Document

from app.vectorstore.vector_repository import VectorRepository


class KeywordRetriever:
    def __init__(self, vector_repository: VectorRepository, top_k: int) -> None:
        self._vector_repository = vector_repository
        self._top_k = top_k

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        scored_documents: list[tuple[Document, float]] = []
        for document in self._vector_repository.get_all_documents():
            content_terms = set(self._tokenize(document.page_content))
            overlap = query_terms.intersection(content_terms)
            if not overlap:
                continue
            score = len(overlap) / len(query_terms)
            scored_documents.append((document, score))

        return sorted(scored_documents, key=lambda item: item[1], reverse=True)[: self._top_k]

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", value.lower())