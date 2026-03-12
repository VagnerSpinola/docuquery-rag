from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, supported_suffixes: set[str] | None = None) -> None:
        self._supported_suffixes = supported_suffixes or {".pdf", ".txt"}

    def load_path(self, source_path: Path) -> list[Document]:
        paths = self._resolve_paths(source_path)
        documents: list[Document] = []

        for path in paths:
            documents.extend(self._load_file(path))

        if not documents:
            raise ValueError(f"No supported documents were found in {source_path}.")

        return documents

    def _resolve_paths(self, source_path: Path) -> list[Path]:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        if source_path.is_file():
            return [source_path] if source_path.suffix.lower() in self._supported_suffixes else []

        return sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in self._supported_suffixes
        )

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded_documents = PyPDFLoader(str(path)).load()
        elif suffix == ".txt":
            loaded_documents = TextLoader(str(path), autodetect_encoding=True).load()
        else:
            return []

        for document in loaded_documents:
            document.metadata["source"] = str(path.resolve())
            document.metadata["document_name"] = path.name

        return loaded_documents