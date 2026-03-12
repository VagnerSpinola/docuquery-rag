from dataclasses import dataclass

from app.agents.reasoning_agent import ReasoningAgent
from app.agents.search_agent import SearchAgent


@dataclass(slots=True)
class OrchestrationPlan:
    retrieval_required: bool
    retrieval_query: str
    refined_question: str
    prompt_instruction: str


class OrchestrationAgent:
    def __init__(self, search_agent: SearchAgent, reasoning_agent: ReasoningAgent) -> None:
        self._search_agent = search_agent
        self._reasoning_agent = reasoning_agent

    def plan(self, question: str) -> OrchestrationPlan:
        search_decision = self._search_agent.decide(question)
        refined_question = self._reasoning_agent.refine_question(question)
        return OrchestrationPlan(
            retrieval_required=search_decision.retrieval_required,
            retrieval_query=search_decision.search_query,
            refined_question=refined_question,
            prompt_instruction=self._reasoning_agent.improve_prompt_instruction(),
        )