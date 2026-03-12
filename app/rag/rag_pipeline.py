from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document

from app.agents.orchestration_agent import OrchestrationAgent
from app.cache.redis_cache import RedisCache
from app.llm.llm_provider import LLMProvider
from app.rag.context_optimizer import ContextOptimizer
from app.rag.prompt_builder import PromptBuilder
from app.retrieval.hybrid_retriever import HybridRetriever


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]


@dataclass(slots=True)
class RAGStreamResponse:
    stream: Iterator[str]
    sources: list[dict[str, Any]]


class RAGPipeline:
    def __init__(
        self,
        llm_provider: LLMProvider,
        retriever: HybridRetriever,
        prompt_builder: PromptBuilder,
        context_optimizer: ContextOptimizer,
        orchestration_agent: OrchestrationAgent,
        cache: RedisCache,
    ) -> None:
        self._llm_provider = llm_provider
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._context_optimizer = context_optimizer
        self._orchestration_agent = orchestration_agent
        self._cache = cache

    def answer(self, question: str) -> RAGResponse:
        cache_key = RedisCache.build_key("response", question)
        cached_response = self._cache.get_json(cache_key)
        if cached_response is not None:
            return RAGResponse(answer=cached_response["answer"], sources=cached_response["sources"])

        plan = self._orchestration_agent.plan(question)
        documents = self._retrieve_documents(plan)
        if not documents:
            return RAGResponse(
                answer="I could not find relevant information in the indexed documents.",
                sources=[],
            )

        optimized_documents = self._context_optimizer.optimize(documents)
        messages = self._prompt_builder.build(plan.refined_question, optimized_documents, plan.prompt_instruction)
        answer = self._llm_provider.generate(messages)
        sources = self._extract_sources(optimized_documents)
        self._cache.set_json(cache_key, {"answer": answer, "sources": sources})
        return RAGResponse(answer=answer, sources=sources)

    def answer_stream(self, question: str) -> RAGStreamResponse:
        plan = self._orchestration_agent.plan(question)
        documents = self._retrieve_documents(plan)
        if not documents:
            return RAGStreamResponse(
                stream=iter(["I could not find relevant information in the indexed documents."]),
                sources=[],
            )

        optimized_documents = self._context_optimizer.optimize(documents)
        messages = self._prompt_builder.build(plan.refined_question, optimized_documents, plan.prompt_instruction)
        return RAGStreamResponse(
            stream=self._llm_provider.stream_generate(messages),
            sources=self._extract_sources(optimized_documents),
        )

    def _retrieve_documents(self, plan: Any) -> list[Document]:
        if not plan.retrieval_required:
            return []

        return self._retriever.retrieve(plan.retrieval_query)

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, Any, Any]] = set()

        for document in documents:
            source = document.metadata.get("document_name") or document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            chunk_id = document.metadata.get("chunk_id")
            key = (str(source), page, chunk_id)

            if key in seen:
                continue

            seen.add(key)
            sources.append(
                {
                    "source": str(source),
                    "page": page,
                    "chunk_id": chunk_id,
                }
            )

        return sources