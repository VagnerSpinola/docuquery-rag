from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.services.document_service import DocumentService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StubIngestionPipeline:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def ingest(self, source_path: Path):
        self.paths.append(source_path)
        return type(
            "IngestionResult",
            (),
            {
                "files_processed": 1,
                "documents_loaded": 1,
                "chunks_created": 3,
            },
        )()


class StubSettings:
    def __init__(self, documents_directory: Path) -> None:
        self.documents_directory = documents_directory


@pytest.mark.anyio
async def test_document_service_uploads_and_ingests_files(tmp_path: Path) -> None:
    service = DocumentService(StubSettings(tmp_path), StubIngestionPipeline())
    upload = UploadFile(filename="policy.txt", file=BytesIO(b"refund policy"))

    result = await service.upload_and_ingest([upload])

    assert result.filenames == ["policy.txt"]
    assert result.files_processed == 1
    assert result.documents_loaded == 1
    assert result.chunks_created == 3
    assert (tmp_path / "policy.txt").exists()


@pytest.mark.anyio
async def test_document_service_rejects_unsupported_file_type(tmp_path: Path) -> None:
    service = DocumentService(StubSettings(tmp_path), StubIngestionPipeline())
    upload = UploadFile(filename="policy.docx", file=BytesIO(b"content"))

    with pytest.raises(ValueError, match="Only PDF and TXT documents are supported"):
        await service.upload_and_ingest([upload])