import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import PlainTextLoader
from app.vectorstore.vector_repository import VectorRepository


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    files_processed: int
    documents_loaded: int
    chunks_created: int


class IngestionPipeline:
    def __init__(
        self,
        pdf_loader: PDFLoader,
        text_loader: PlainTextLoader,
        semantic_chunker: SemanticChunker,
        vector_repository: VectorRepository,
    ) -> None:
        self._pdf_loader = pdf_loader
        self._text_loader = text_loader
        self._semantic_chunker = semantic_chunker
        self._vector_repository = vector_repository

    def ingest(self, source_path: Path, reset_collection: bool = False) -> IngestionResult:
        if reset_collection:
            self._vector_repository.reset_collection()

        raw_documents = self._load_documents(source_path)
        chunked_documents = self._semantic_chunker.split(raw_documents)
        self._decorate_chunk_metadata(chunked_documents)
        self._vector_repository.add_documents(chunked_documents)

        files_processed = len({document.metadata.get("source") for document in raw_documents})
        logger.info(
            "Ingestion completed with %s files, %s raw documents, and %s chunks.",
            files_processed,
            len(raw_documents),
            len(chunked_documents),
        )

        return IngestionResult(
            files_processed=files_processed,
            documents_loaded=len(raw_documents),
            chunks_created=len(chunked_documents),
        )

    def _load_documents(self, source_path: Path) -> list[Document]:
        paths = self._resolve_paths(source_path)
        documents: list[Document] = []

        for path in paths:
            if path.suffix.lower() == ".pdf":
                documents.extend(self._pdf_loader.load(path))
            elif path.suffix.lower() == ".txt":
                documents.extend(self._text_loader.load(path))

        if not documents:
            raise ValueError(f"No supported documents were found in {source_path}.")

        return documents

    @staticmethod
    def _resolve_paths(source_path: Path) -> list[Path]:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        if source_path.is_file():
            return [source_path]

        return sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
        )

    @staticmethod
    def _decorate_chunk_metadata(documents: list[Document]) -> None:
        for index, document in enumerate(documents):
            page = document.metadata.get("page")
            if isinstance(page, int) and page >= 0:
                document.metadata["page"] = page + 1

            source = document.metadata.get("document_name") or document.metadata.get("source", "unknown")
            page_marker = document.metadata.get("page", "na")
            document.metadata["chunk_id"] = f"{source}:{page_marker}:{index}"