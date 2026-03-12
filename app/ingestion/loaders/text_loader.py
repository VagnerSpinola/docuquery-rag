from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class PlainTextLoader:
    def load(self, path: Path) -> list[Document]:
        documents = TextLoader(str(path), autodetect_encoding=True).load()
        for document in documents:
            document.metadata["source"] = str(path.resolve())
            document.metadata["document_name"] = path.name

        return documents