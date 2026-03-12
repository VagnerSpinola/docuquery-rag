import logging
from typing import Iterable

from chromadb import PersistentClient
from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.config import Settings
from app.embeddings.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class VectorRepository:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self._settings = settings
        self._embedding_function = embedding_service.get_embedding_model()
        self._client = PersistentClient(path=str(settings.chroma_persist_directory))
        self._store = self._build_store()

    def _build_store(self) -> Chroma:
        return Chroma(
            collection_name=self._settings.chroma_collection_name,
            persist_directory=str(self._settings.chroma_persist_directory),
            embedding_function=self._embedding_function,
        )

    def add_documents(self, documents: Iterable[Document]) -> None:
        documents = list(documents)
        if not documents:
            logger.warning("No documents received for vector store ingestion.")
            return

        ids = [str(document.metadata["chunk_id"]) for document in documents]
        self._store.add_documents(documents=documents, ids=ids)
        logger.info("Persisted %s document chunks into Chroma.", len(documents))

    def similarity_search(self, query: str, top_k: int) -> list[Document]:
        return self._store.similarity_search(query=query, k=top_k)

    def similarity_search_with_scores(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query=query, k=top_k)

    def get_all_documents(self) -> list[Document]:
        if not self.collection_exists():
            return []

        collection = self._client.get_collection(self._settings.chroma_collection_name)
        payload = collection.get(include=["documents", "metadatas"])

        documents: list[Document] = []
        for content, metadata in zip(payload.get("documents", []), payload.get("metadatas", []), strict=False):
            if content is None:
                continue
            documents.append(Document(page_content=content, metadata=metadata or {}))

        return documents

    def collection_exists(self) -> bool:
        collections = self._client.list_collections()
        collection_names = {getattr(collection, "name", collection) for collection in collections}
        return self._settings.chroma_collection_name in collection_names

    def reset_collection(self) -> None:
        collections = self._client.list_collections()
        collection_names = {getattr(collection, "name", collection) for collection in collections}

        if self._settings.chroma_collection_name in collection_names:
            self._client.delete_collection(self._settings.chroma_collection_name)
            logger.info(
                "Deleted existing Chroma collection '%s'.",
                self._settings.chroma_collection_name,
            )

        self._store = self._build_store()