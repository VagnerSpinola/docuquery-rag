from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


class PromptBuilder:
    def build(
        self,
        question: str,
        documents: list[Document],
        prompt_instruction: str = "",
    ) -> list[SystemMessage | HumanMessage]:
        context_sections = []

        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("document_name") or document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            page_label = f"Page: {page}" if page else "Page: n/a"
            context_sections.append(
                f"[Chunk {index}]\nSource: {source}\n{page_label}\nContent:\n{document.page_content}"
            )

        system_prompt = (
            "You are Neural Knowledge Engine, an enterprise retrieval-augmented assistant. "
            "Answer only from the provided context. If the answer is not supported by the context, "
            "say that the information is not available in the indexed documents. "
            "Be concise, factual, and avoid fabrication. "
            f"{prompt_instruction}".strip()
        )
        human_prompt = (
            "Use the retrieved document context below to answer the question.\n\n"
            f"Context:\n{chr(10).join(context_sections)}\n\n"
            f"Question: {question}\n\n"
            "Return a direct answer grounded in the context."
        )

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]