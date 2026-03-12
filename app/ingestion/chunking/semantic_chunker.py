from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticChunker:
    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)