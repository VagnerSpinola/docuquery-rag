import argparse
import json
import logging
from pathlib import Path

from app.agents.orchestration_agent import OrchestrationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.search_agent import SearchAgent
from app.cache.redis_cache import RedisCache
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.evaluation_pipeline import EvaluationPipeline
from app.evaluation.rag_metrics import RAGMetrics
from app.llm.llm_provider import LLMProvider
from app.rag.context_optimizer import ContextOptimizer
from app.rag.prompt_builder import PromptBuilder
from app.rag.rag_pipeline import RAGPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.services.chat_service import ChatService
from app.vectorstore.vector_repository import VectorRepository


logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a JSON dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the evaluation dataset JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write evaluation results.")
    return parser


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json_format)
    args = build_argument_parser().parse_args()

    cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds)
    embedding_service = EmbeddingService(settings, cache)
    vector_repository = VectorRepository(settings, embedding_service)
    rag_pipeline = RAGPipeline(
        llm_provider=LLMProvider(settings),
        retriever=HybridRetriever(
            vector_retriever=VectorRetriever(vector_repository, settings.retrieval_k),
            keyword_retriever=KeywordRetriever(vector_repository, settings.retrieval_k),
            cache=cache,
            top_k=settings.retrieval_k,
            vector_weight=settings.hybrid_vector_weight,
            keyword_weight=settings.hybrid_keyword_weight,
        ),
        prompt_builder=PromptBuilder(),
        context_optimizer=ContextOptimizer(settings.context_max_documents, settings.context_max_characters),
        orchestration_agent=OrchestrationAgent(SearchAgent(), ReasoningAgent()),
        cache=cache,
    )
    pipeline = EvaluationPipeline(ChatService(rag_pipeline), RAGMetrics())
    report = pipeline.evaluate_file(args.dataset)

    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
        logger.info("Evaluation report written to %s", args.output)
    else:
        print(output)


if __name__ == "__main__":
    main()