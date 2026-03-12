from collections.abc import Iterator
from typing import Protocol


class RAGPipelineContract(Protocol):
    def answer(self, question: str) -> object:
        ...

    def answer_stream(self, question: str) -> object:
        ...


class ChatService:
    def __init__(self, rag_pipeline: RAGPipelineContract) -> None:
        self._rag_pipeline = rag_pipeline

    def ask(self, question: str) -> dict[str, object]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        response = self._rag_pipeline.answer(normalized_question)
        return {
            "answer": response.answer,
            "sources": response.sources,
        }

    def ask_stream(self, question: str) -> tuple[Iterator[str], list[dict[str, object]]]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        response = self._rag_pipeline.answer_stream(normalized_question)
        return response.stream, response.sources