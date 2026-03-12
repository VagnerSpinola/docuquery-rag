from pathlib import Path


def test_ingest_documents_task_dispatches_via_celery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from workers import ingestion_worker

    class StubIngestionPipeline:
        def ingest(self, source_path: Path, reset_collection: bool = False):
            assert source_path == tmp_path
            assert reset_collection is True
            return type(
                "IngestionResult",
                (),
                {
                    "files_processed": 2,
                    "documents_loaded": 4,
                    "chunks_created": 8,
                },
            )()

    class StubEvaluationPipeline:
        def evaluate_file(self, dataset_path: Path):
            return {}

    monkeypatch.setattr(
        ingestion_worker,
        "_build_runtime",
        lambda: (StubIngestionPipeline(), StubEvaluationPipeline()),
    )
    ingestion_worker.celery_app.conf.task_always_eager = True
    ingestion_worker.celery_app.conf.task_store_eager_result = False
    ingestion_worker.celery_app.conf.broker_url = "memory://"
    ingestion_worker.celery_app.conf.result_backend = "cache+memory://"

    result = ingestion_worker.ingest_documents_task.delay(str(tmp_path), True)

    assert result.successful()
    assert result.get() == {
        "files_processed": 2,
        "documents_loaded": 4,
        "chunks_created": 8,
    }


def test_evaluate_rag_task_dispatches_via_celery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from workers import ingestion_worker

    class StubIngestionPipeline:
        def ingest(self, source_path: Path, reset_collection: bool = False):
            return None

    class StubEvaluationPipeline:
        def evaluate_file(self, dataset_path: Path):
            assert dataset_path == tmp_path
            return {"aggregates": {"answer_relevance": 0.9}}

    monkeypatch.setattr(
        ingestion_worker,
        "_build_runtime",
        lambda: (StubIngestionPipeline(), StubEvaluationPipeline()),
    )
    ingestion_worker.celery_app.conf.task_always_eager = True
    ingestion_worker.celery_app.conf.task_store_eager_result = False
    ingestion_worker.celery_app.conf.broker_url = "memory://"
    ingestion_worker.celery_app.conf.result_backend = "cache+memory://"

    result = ingestion_worker.evaluate_rag_task.delay(str(tmp_path))

    assert result.successful()
    assert result.get() == {"aggregates": {"answer_relevance": 0.9}}