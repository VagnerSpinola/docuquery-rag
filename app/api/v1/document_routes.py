from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.services.document_service import DocumentService


router = APIRouter(tags=["documents"])


class UploadResponse(BaseModel):
    filenames: list[str] = Field(default_factory=list)
    files_processed: int
    documents_loaded: int
    chunks_created: int


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


@router.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: list[UploadFile] = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    try:
        result = await document_service.upload_and_ingest(files)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return UploadResponse(
        filenames=result.filenames,
        files_processed=result.files_processed,
        documents_loaded=result.documents_loaded,
        chunks_created=result.chunks_created,
    )