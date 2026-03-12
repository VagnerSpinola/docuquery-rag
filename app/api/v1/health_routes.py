from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter(tags=["health"])


class DependencyStatus(BaseModel):
    status: str
    details: dict[str, bool]


@router.get("/health", response_model=DependencyStatus)
async def health(request: Request) -> DependencyStatus:
    cache = request.app.state.cache
    metadata_store = request.app.state.metadata_store
    vector_repository = request.app.state.vector_repository
    settings = request.app.state.settings

    details = {
        "redis": cache.ping(),
        "postgres": metadata_store.ping() if metadata_store is not None else False,
        "chroma": settings.chroma_persist_directory.exists() and vector_repository is not None,
        "celery": bool(request.app.state.settings.celery_broker_url),
    }
    overall = "healthy" if all(details.values()) else "degraded"
    return DependencyStatus(status=overall, details=details)