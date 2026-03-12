from dataclasses import dataclass


@dataclass(slots=True)
class SearchDecision:
    retrieval_required: bool
    search_query: str
    reasoning: str


class SearchAgent:
    def decide(self, question: str) -> SearchDecision:
        normalized = question.strip()
        retrieval_required = not normalized.lower().startswith(("hi", "hello", "hey"))
        reasoning = "Retrieve enterprise knowledge base context." if retrieval_required else "Use direct response."
        return SearchDecision(
            retrieval_required=retrieval_required,
            search_query=normalized,
            reasoning=reasoning,
        )