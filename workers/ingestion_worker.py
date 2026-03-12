from pathlib import Path

from celery import Celery

from app.agents.orchestration_agent import OrchestrationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.search_agent import SearchAgent
from app.cache.redis_cache import RedisCache
from app.core.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.evaluation_pipeline import EvaluationPipeline
from app.evaluation.rag_metrics import RAGMetrics
from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import PlainTextLoader
from app.llm.llm_provider import LLMProvider
from app.rag.context_optimizer import ContextOptimizer
from app.rag.prompt_builder import PromptBuilder
from app.rag.rag_pipeline import RAGPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.services.chat_service import ChatService
from app.vectorstore.vector_repository import VectorRepository


settings = get_settings()
celery_app = Celery(
    "neural_knowledge_engine",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


def _build_runtime() -> tuple[IngestionPipeline, EvaluationPipeline]:
    cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds)
    embedding_service = EmbeddingService(settings, cache)
    vector_repository = VectorRepository(settings, embedding_service)

    ingestion_pipeline = IngestionPipeline(
        pdf_loader=PDFLoader(),
        text_loader=PlainTextLoader(),
        semantic_chunker=SemanticChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens),
        vector_repository=vector_repository,
    )

    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(vector_repository, settings.retrieval_k),
        keyword_retriever=KeywordRetriever(vector_repository, settings.retrieval_k),
        cache=cache,
        top_k=settings.retrieval_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_weight=settings.hybrid_keyword_weight,
    )
    rag_pipeline = RAGPipeline(
        llm_provider=LLMProvider(settings),
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        context_optimizer=ContextOptimizer(settings.context_max_documents, settings.context_max_characters),
        orchestration_agent=OrchestrationAgent(SearchAgent(), ReasoningAgent()),
        cache=cache,
    )
    evaluation_pipeline = EvaluationPipeline(ChatService(rag_pipeline), RAGMetrics())
    return ingestion_pipeline, evaluation_pipeline


@celery_app.task(name="workers.ingestion_worker.ingest_documents")
def ingest_documents_task(path: str, reset_collection: bool = False) -> dict[str, int]:
    ingestion_pipeline, _ = _build_runtime()
    result = ingestion_pipeline.ingest(Path(path), reset_collection=reset_collection)
    return {
        "files_processed": result.files_processed,
        "documents_loaded": result.documents_loaded,
        "chunks_created": result.chunks_created,
    }


@celery_app.task(name="workers.ingestion_worker.evaluate_rag")
def evaluate_rag_task(dataset_path: str) -> dict[str, object]:
    _, evaluation_pipeline = _build_runtime()
    return evaluation_pipeline.evaluate_file(Path(dataset_path))