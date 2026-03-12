from app.evaluation.rag_metrics import RAGMetrics


def test_rag_metrics_return_bounded_scores() -> None:
    metrics = RAGMetrics()

    answer_relevance = metrics.answer_relevance("What is the refund policy?", "The refund policy allows returns.")
    faithfulness = metrics.faithfulness("refund policy allows returns", ["The refund policy allows returns within 30 days."])
    context_precision = metrics.context_precision("refund policy allows returns", ["The refund policy allows returns within 30 days."])
    retrieval_recall = metrics.retrieval_recall(["policy.txt"], ["policy.txt", "handbook.txt"])

    for value in [answer_relevance, faithfulness, context_precision, retrieval_recall]:
        assert 0.0 <= value <= 1.0