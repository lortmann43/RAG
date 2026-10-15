from pathlib import Path
import sys
from langchain_core.documents import Document
from tqdm import tqdm
from .chunkers import Chunker
import bm25s
import json


class Indexer:
    """Discover source files and build a searchable index."""

    SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}

    def __init__(
            self,
            source_directory: Path,
            max_chunk_size: int = 2000) -> None:
        self.source_directory = source_directory
        self.chunker = Chunker(max_chunk_size)
        self.chunks: list[Document] = []
        self.bm25: bm25s.BM25 | None = None

    def discover_files(self) -> list[Path]:
        """Find all supported files in the source directory"""
        if not self.source_directory.is_dir():
            raise FileNotFoundError(
                f"Source directory not found: {self.source_directory}")

        files = [
            path for path in self.source_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        return sorted(files)

    def collect_chunks(self) -> list[Document]:
        """Split all discovered files into chunks."""
        files = self.discover_files()
        chunks: list[Document] = []

        for path in tqdm(files, desc="Chunking", unit="File"):
            try:
                file_chunks = self.chunker.split_file(path)
                chunks.extend(file_chunks)
            except (OSError, UnicodeError) as exc:
                print(f"Skipping {path}: {exc}", file=sys.stderr)

        return chunks

    def build_index(self) -> None:
        """Build a BM25 index from all source chunks."""
        self.chunks = self.collect_chunks()

        if not self.chunks:
            raise ValueError("No chunks available for indexing!")

        corpus = [chunk.page_content for chunk in self.chunks]

        print("Tokenizing chunks...")
        tokenized_corpus = bm25s.tokenize(corpus)

        print("Building BM25 index...")
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokenized_corpus)

        print(f"Indexed {len(self.chunks)} chunks successfully!")

    def save_index(self, directory: Path) -> None:
        """Save the BM25 index and source chunks."""
        if self.bm25 is None or not self.chunks:
            raise ValueError("Build the index before saving")

        directory.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(directory / "bm25"))
        chunks_path = directory / "chunks.jsonl"

        with chunks_path.open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                data = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_index(self, directory: Path) -> None:
        """Load the BM25 index and source chunks."""
        self.bm25 = bm25s.BM25.load(
            str(directory / "bm25"),
            load_corpus=False
        )

        chunks_path = directory / "chunks.jsonl"

        with chunks_path.open("r", encoding="utf-8") as file:
            self.chunks = [
                Document(**json.loads(line))
                for line in file
            ]
