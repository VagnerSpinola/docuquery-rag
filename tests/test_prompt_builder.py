from langchain_core.documents import Document

from app.rag.prompt_builder import PromptBuilder


def test_prompt_builder_includes_question_and_context() -> None:
    builder = PromptBuilder()
    documents = [
        Document(
            page_content="Refunds are available within 30 days of purchase.",
            metadata={"document_name": "policy.pdf", "page": 2, "chunk_id": "policy.pdf:2:0"},
        )
    ]

    messages = builder.build("What is the refund policy?", documents, "Use precise enterprise language.")

    assert len(messages) == 2
    assert "What is the refund policy?" in messages[1].content
    assert "Refunds are available within 30 days of purchase." in messages[1].content
    assert "policy.pdf" in messages[1].content
    assert "Use precise enterprise language." in messages[0].content