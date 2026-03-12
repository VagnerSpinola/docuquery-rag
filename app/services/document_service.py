import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine, func, insert, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.ingestion.ingestion_pipeline import IngestionPipeline, IngestionResult


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UploadResult:
    filenames: list[str]
    files_processed: int
    documents_loaded: int
    chunks_created: int


class SQLDocumentMetadataStore:
    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, future=True, pool_pre_ping=True)
        self._metadata = MetaData()
        self._documents = Table(
            "documents",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("document_name", String(255), nullable=False),
            Column("source_path", String(1024), nullable=False),
            Column("status", String(64), nullable=False),
            Column("file_size_bytes", Integer, nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        )

    def ensure_schema(self) -> None:
        try:
            self._metadata.create_all(self._engine)
        except SQLAlchemyError:
            logger.warning("PostgreSQL metadata schema creation failed.", exc_info=True)

    def save_document(self, document_name: str, source_path: Path, file_size_bytes: int, status: str) -> None:
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    insert(self._documents).values(
                        document_name=document_name,
                        source_path=str(source_path),
                        status=status,
                        file_size_bytes=file_size_bytes,
                    )
                )
        except SQLAlchemyError:
            logger.warning("Failed to persist document metadata for %s.", document_name, exc_info=True)

    def ping(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        ingestion_pipeline: IngestionPipeline,
        metadata_store: SQLDocumentMetadataStore | None = None,
    ) -> None:
        self._settings = settings
        self._ingestion_pipeline = ingestion_pipeline
        self._metadata_store = metadata_store
        self._allowed_suffixes = {".pdf", ".txt"}

    async def upload_and_ingest(self, files: list[UploadFile]) -> UploadResult:
        if not files:
            raise ValueError("At least one document must be provided.")

        saved_paths: list[Path] = []
        saved_names: list[str] = []
        totals = IngestionResult(files_processed=0, documents_loaded=0, chunks_created=0)

        for upload in files:
            saved_path, file_size_bytes = await self._save_upload(upload)
            saved_paths.append(saved_path)
            saved_names.append(saved_path.name)

            result = self._ingestion_pipeline.ingest(saved_path)
            if self._metadata_store is not None:
                self._metadata_store.save_document(saved_path.name, saved_path, file_size_bytes, status="ingested")
            totals = IngestionResult(
                files_processed=totals.files_processed + result.files_processed,
                documents_loaded=totals.documents_loaded + result.documents_loaded,
                chunks_created=totals.chunks_created + result.chunks_created,
            )

        logger.info(
            "Uploaded and ingested documents.",
            extra={
                "filenames": saved_names,
                "files_processed": totals.files_processed,
                "chunks_created": totals.chunks_created,
            },
        )

        return UploadResult(
            filenames=saved_names,
            files_processed=totals.files_processed,
            documents_loaded=totals.documents_loaded,
            chunks_created=totals.chunks_created,
        )

    async def _save_upload(self, upload: UploadFile) -> tuple[Path, int]:
        filename = Path(upload.filename or "").name
        if not filename:
            raise ValueError("Uploaded files must include a filename.")

        suffix = Path(filename).suffix.lower()
        if suffix not in self._allowed_suffixes:
            raise ValueError("Only PDF and TXT documents are supported.")

        target_path = self._build_target_path(filename)
        content = await upload.read()
        target_path.write_bytes(content)
        await upload.close()
        return target_path, len(content)

    def _build_target_path(self, filename: str) -> Path:
        target_path = self._settings.documents_directory / filename
        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            target_path = self._settings.documents_directory / f"{stem}-{uuid4().hex[:8]}{suffix}"

        return target_path