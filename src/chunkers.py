
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter
)


class Chunker:
    """Split source files into chunks while preserving offsets."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        if not 1 <= max_chunk_size <= 2000:
            raise ValueError("Chunk size must be between 1 and 2000")

        overlap = min(200, max_chunk_size // 10)

        self.python_splitter = (
            RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=max_chunk_size,
                chunk_overlap=overlap,
                add_start_index=True,
                strip_whitespace=False
            )
        )

        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
            strip_whitespace=False
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
            strip_whitespace=False
        )

    def split_python(self, text: str) -> list[Document]:
        """Split Python source code."""
        return self.python_splitter.create_documents([text])

    def split_markdown(self, text: str) -> list[Document]:
        """Split Markdown documentation."""
        return self.markdown_splitter.create_documents([text])

    def split_text(self, text: str) -> list[Document]:
        """Split plain text."""
        return self.text_splitter.create_documents([text])

    def split_file(self, file_path: Path) -> list[Document]:
        """Read a file and select its chunking strategy."""
        suffix = file_path.suffix.lower()

        if suffix not in {".py", ".md", ".txt"}:
            raise ValueError(f"Unsupported file type: {suffix}")

        with file_path.open("r", encoding="utf-8", newline="") as file:
            text = file.read()

        if suffix == ".py":
            chunks = self.split_python(text)
        elif suffix == ".md":
            chunks = self.split_markdown(text)
        else:
            chunks = self.split_text(text)

        for chunk in chunks:
            chunk.metadata["file_path"] = file_path.as_posix()

        return chunks
