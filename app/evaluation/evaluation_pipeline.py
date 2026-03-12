import json
from pathlib import Path

from app.evaluation.rag_metrics import RAGMetrics
from app.services.chat_service import ChatService


class EvaluationPipeline:
    def __init__(self, chat_service: ChatService, metrics: RAGMetrics) -> None:
        self._chat_service = chat_service
        self._metrics = metrics

    def evaluate_file(self, dataset_path: Path) -> dict[str, object]:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        results = []

        for item in dataset:
            response = self._chat_service.ask(item["question"])
            contexts = [source.get("source", "") for source in response["sources"]]
            metrics = {
                "question": item["question"],
                "answer_relevance": self._metrics.answer_relevance(item["question"], str(response["answer"])),
                "faithfulness": self._metrics.faithfulness(str(response["answer"]), contexts),
                "context_precision": self._metrics.context_precision(str(response["answer"]), contexts),
                "retrieval_recall": self._metrics.retrieval_recall(
                    retrieved_sources=contexts,
                    expected_sources=item.get("expected_sources", []),
                ),
            }
            results.append(metrics)

        aggregates = {
            metric: sum(result[metric] for result in results) / len(results) if results else 0.0
            for metric in ["answer_relevance", "faithfulness", "context_precision", "retrieval_recall"]
        }
        return {"results": results, "aggregates": aggregates}