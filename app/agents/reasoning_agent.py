class ReasoningAgent:
    def refine_question(self, question: str) -> str:
        return question.strip().replace("\n", " ")

    def improve_prompt_instruction(self) -> str:
        return (
            "Prioritize factual correctness, cite the most relevant sources, and say explicitly when the evidence is insufficient."
        )