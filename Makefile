PYTHON ?= python
UVICORN ?= uvicorn
APP_MODULE ?= app.main:app
DOCS_PATH ?= data/documents
EVAL_DATASET ?= data/evaluation/rag_eval.json

.PHONY: install test run ingest ingest-reset evaluate worker docker-up docker-down

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

run:
	$(UVICORN) $(APP_MODULE) --host 0.0.0.0 --port 8000 --reload

ingest:
	$(PYTHON) -m scripts.ingest_documents --path $(DOCS_PATH)

ingest-reset:
	$(PYTHON) -m scripts.ingest_documents --path $(DOCS_PATH) --reset

evaluate:
	$(PYTHON) -m scripts.evaluate_rag --dataset $(EVAL_DATASET)

worker:
	celery -A workers.ingestion_worker.celery_app worker --loglevel=info

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down