import re
from collections.abc import Iterable


def _tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", value.lower()))


class RAGMetrics:
    def answer_relevance(self, question: str, answer: str) -> float:
        question_tokens = _tokenize(question)
        answer_tokens = _tokenize(answer)
        if not question_tokens:
            return 0.0
        return len(question_tokens.intersection(answer_tokens)) / len(question_tokens)

    def faithfulness(self, answer: str, contexts: Iterable[str]) -> float:
        context_tokens = set().union(*(_tokenize(context) for context in contexts)) if contexts else set()
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        return len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)

    def context_precision(self, answer: str, contexts: Iterable[str]) -> float:
        answer_tokens = _tokenize(answer)
        context_tokens = set().union(*(_tokenize(context) for context in contexts)) if contexts else set()
        if not context_tokens:
            return 0.0
        return len(answer_tokens.intersection(context_tokens)) / len(context_tokens)

    def retrieval_recall(self, retrieved_sources: Iterable[str], expected_sources: Iterable[str]) -> float:
        expected = set(expected_sources)
        retrieved = set(retrieved_sources)
        if not expected:
            return 0.0
        return len(expected.intersection(retrieved)) / len(expected)