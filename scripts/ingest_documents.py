import argparse
import logging
from pathlib import Path

from app.cache.redis_cache import RedisCache
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import PlainTextLoader
from app.vectorstore.vector_repository import VectorRepository


logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDF and TXT documents into ChromaDB.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="File or directory to ingest. Defaults to DOCUMENTS_DIRECTORY from configuration.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector collection before ingestion.",
    )
    return parser


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json_format)
    args = build_argument_parser().parse_args()

    source_path = args.path or settings.documents_directory
    cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds)
    embedding_service = EmbeddingService(settings, cache)
    vector_repository = VectorRepository(settings, embedding_service)
    ingestion_pipeline = IngestionPipeline(
        pdf_loader=PDFLoader(),
        text_loader=PlainTextLoader(),
        semantic_chunker=SemanticChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens),
        vector_repository=vector_repository,
    )

    result = ingestion_pipeline.ingest(source_path=source_path, reset_collection=args.reset)
    logger.info(
        "Ingested %s files, %s raw documents, and %s chunks from %s.",
        result.files_processed,
        result.documents_loaded,
        result.chunks_created,
        source_path,
    )


if __name__ == "__main__":
    main()