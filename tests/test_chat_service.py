import pytest

from app.services.chat_service import ChatService


class StubRAGPipeline:
    def answer(self, question: str):
        return type(
            "Response",
            (),
            {
                "answer": f"Echo: {question}",
                "sources": [{"source": "handbook.txt", "page": 1, "chunk_id": "chunk-1"}],
            },
        )()


def test_chat_service_returns_answer_and_sources() -> None:
    service = ChatService(StubRAGPipeline())

    response = service.ask("What is the onboarding policy?")

    assert response["answer"] == "Echo: What is the onboarding policy?"
    assert response["sources"] == [{"source": "handbook.txt", "page": 1, "chunk_id": "chunk-1"}]


def test_chat_service_rejects_blank_questions() -> None:
    service = ChatService(StubRAGPipeline())

    with pytest.raises(ValueError, match="Question must not be empty"):
        service.ask("   ")