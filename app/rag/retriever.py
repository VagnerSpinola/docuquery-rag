from langchain_core.documents import Document

from app.vectorstore.vector_repository import VectorRepository


class Retriever:
    def __init__(self, vector_repository: VectorRepository, top_k: int) -> None:
        self._retriever = vector_repository.as_retriever(top_k)

    def retrieve(self, question: str) -> list[Document]:
        return self._retriever.invoke(question)