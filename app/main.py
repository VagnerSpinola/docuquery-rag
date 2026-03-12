from contextlib import asynccontextmanager
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.chat_routes import router as chat_router
from app.api.v1.document_routes import router as document_router
from app.api.v1.health_routes import router as health_router
from app.agents.orchestration_agent import OrchestrationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.search_agent import SearchAgent
from app.cache.redis_cache import RedisCache
from app.core.config import get_settings
from app.core.logging import configure_logging, reset_request_id, set_request_id
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
from app.services.document_service import DocumentService, SQLDocumentMetadataStore
from app.vectorstore.vector_repository import VectorRepository


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json_format)

    cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds)
    embedding_service = EmbeddingService(settings, cache)
    vector_repository = VectorRepository(settings, embedding_service)
    metadata_store = SQLDocumentMetadataStore(settings.database_url)
    metadata_store.ensure_schema()
    ingestion_pipeline = IngestionPipeline(
        pdf_loader=PDFLoader(),
        text_loader=PlainTextLoader(),
        semantic_chunker=SemanticChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens),
        vector_repository=vector_repository,
    )
    llm_provider = LLMProvider(settings)
    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(vector_repository, settings.retrieval_k),
        keyword_retriever=KeywordRetriever(vector_repository, settings.retrieval_k),
        cache=cache,
        top_k=settings.retrieval_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_weight=settings.hybrid_keyword_weight,
    )
    prompt_builder = PromptBuilder()
    rag_pipeline = RAGPipeline(
        llm_provider=llm_provider,
        retriever=retriever,
        prompt_builder=prompt_builder,
        context_optimizer=ContextOptimizer(settings.context_max_documents, settings.context_max_characters),
        orchestration_agent=OrchestrationAgent(SearchAgent(), ReasoningAgent()),
        cache=cache,
    )
    chat_service = ChatService(rag_pipeline)
    document_service = DocumentService(settings, ingestion_pipeline, metadata_store)

    app.state.settings = settings
    app.state.cache = cache
    app.state.vector_repository = vector_repository
    app.state.metadata_store = metadata_store
    app.state.chat_service = chat_service
    app.state.document_service = document_service
    app.state.evaluation_pipeline = EvaluationPipeline(chat_service, RAGMetrics())
    logger.info("Application startup completed for %s.", settings.app_name)
    yield
    logger.info("Application shutdown completed.")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def add_request_context(request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        token = set_request_id(request_id)
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "Unhandled request failure.",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            raise
        finally:
            if "response" in locals():
                response.headers["X-Request-ID"] = request_id
                duration_ms = round((perf_counter() - started_at) * 1000, 2)
                logger.info(
                    "Request completed.",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                    },
                )
            reset_request_id(token)

    application.include_router(chat_router, prefix=settings.api_prefix)
    application.include_router(document_router, prefix=settings.api_prefix)
    application.include_router(health_router, prefix=settings.api_prefix)

    if settings.metrics_enabled:
        Instrumentator().instrument(application).expose(application)

    return application


app = create_app()