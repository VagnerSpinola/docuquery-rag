from langchain_core.documents import Document


class ContextOptimizer:
    def __init__(self, max_documents: int, max_characters: int) -> None:
        self._max_documents = max_documents
        self._max_characters = max_characters

    def optimize(self, documents: list[Document]) -> list[Document]:
        selected: list[Document] = []
        consumed = 0
        seen: set[str] = set()

        for document in documents:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if chunk_id in seen:
                continue

            projected = consumed + len(document.page_content)
            if projected > self._max_characters and selected:
                break

            selected.append(document)
            seen.add(chunk_id)
            consumed = projected
            if len(selected) >= self._max_documents:
                break

        return selected