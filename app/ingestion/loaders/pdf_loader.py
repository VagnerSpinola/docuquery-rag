from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    def load(self, path: Path) -> list[Document]:
        documents = PyPDFLoader(str(path)).load()
        for document in documents:
            document.metadata["source"] = str(path.resolve())
            document.metadata["document_name"] = path.name

        return documents