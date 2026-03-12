from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.chat_routes import router as chat_router
from app.api.v1.document_routes import router as document_router
from app.api.v1.health_routes import router as health_router


class StubChatService:
    def ask(self, question: str) -> dict[str, object]:
        return {
            "answer": f"Answer for: {question}",
            "sources": [{"source": "policy.txt", "page": 1, "chunk_id": "chunk-1"}],
        }

    def ask_stream(self, question: str) -> tuple[Iterator[str], list[dict[str, object]]]:
        return iter(["partial-1", "partial-2"]), [
            {"source": "policy.txt", "page": 1, "chunk_id": "chunk-1"}
        ]


class StubDocumentService:
    async def upload_and_ingest(self, files):
        filenames = [file.filename for file in files]
        return SimpleNamespace(
            filenames=filenames,
            files_processed=len(filenames),
            documents_loaded=len(filenames),
            chunks_created=len(filenames) * 2,
        )


class StubCache:
    def __init__(self, status: bool) -> None:
        self._status = status

    def ping(self) -> bool:
        return self._status


class StubMetadataStore:
    def __init__(self, status: bool) -> None:
        self._status = status

    def ping(self) -> bool:
        return self._status


def build_test_app(*, healthy: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(document_router)
    app.include_router(health_router)
    app.state.chat_service = StubChatService()
    app.state.document_service = StubDocumentService()
    app.state.cache = StubCache(healthy)
    app.state.metadata_store = StubMetadataStore(healthy)
    app.state.vector_repository = object()
    app.state.settings = SimpleNamespace(
        chroma_persist_directory=Path.cwd(),
        celery_broker_url="redis://redis:6379/1",
    )
    return app


def test_chat_route_returns_answer_and_sources() -> None:
    client = TestClient(build_test_app())

    response = client.post("/chat", json={"question": "What is the refund policy?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer for: What is the refund policy?",
        "sources": [{"source": "policy.txt", "page": 1, "chunk_id": "chunk-1"}],
    }


def test_chat_stream_route_returns_sse_payload() -> None:
    client = TestClient(build_test_app())

    with client.stream("POST", "/chat/stream", json={"question": "Stream this"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type":"chunk","content":"partial-1"' in body
    assert '"type":"chunk","content":"partial-2"' in body
    assert '"type":"sources"' in body
    assert '"type": "done"' in body


def test_document_upload_route_returns_ingestion_summary() -> None:
    client = TestClient(build_test_app())

    response = client.post(
        "/documents/upload",
        files=[("files", ("policy.txt", b"refund policy", "text/plain"))],
    )

    assert response.status_code == 201
    assert response.json() == {
        "filenames": ["policy.txt"],
        "files_processed": 1,
        "documents_loaded": 1,
        "chunks_created": 2,
    }


def test_health_route_reports_healthy_dependencies() -> None:
    client = TestClient(build_test_app(healthy=True))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "details": {
            "redis": True,
            "postgres": True,
            "chroma": True,
            "celery": True,
        },
    }


def test_health_route_reports_degraded_dependencies() -> None:
    client = TestClient(build_test_app(healthy=False))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["details"]["redis"] is False
    assert response.json()["details"]["postgres"] is False