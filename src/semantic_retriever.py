
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .models.minimal_source_model import MinimalSource


class SemanticRetriever:
    """Index and retrieve chunks using semantic embeddings."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, index_directory: Path) -> None:
        self.index_directory = index_directory
        self.semantic_directory = index_directory / "semantic"

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device="cpu",
            local_files_only=True,
        )

        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def _load_chunks(self) -> list[dict[str, Any]]:
        """Load the chunks produced by the BM25 indexer."""
        chunks_path = self.index_directory / "chunks.jsonl"
        chunks: list[dict[str, Any]] = []

        with chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                chunks.append(json.loads(line))

        return chunks

    def _chunk_fingerprint(self) -> str:
        """Hash the chunk file to detect stale embeddings."""
        path = self.index_directory / "chunks.jsonl"

        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    def build_index(self, batch_size: int = 32) -> None:
        """Generate and persist embeddings for all chunks."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.chunks = self._load_chunks()

        if not self.chunks:
            raise ValueError("No chunks available")

        texts = [
            chunk["page_content"]
            for chunk in self.chunks
        ]

        print(f"Encoding {len(texts)} chunks...")

        self.embeddings = np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

        self.semantic_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            self.semantic_directory / "embeddings.npy",
            self.embeddings,
        )

        manifest = {
            "model": self.MODEL_NAME,
            "chunks_sha256": self._chunk_fingerprint(),
        }

        (self.semantic_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print("Semantic index saved successfully")

    def load_index(self) -> None:
        """Load embeddings and verify the indexed corpus."""
        manifest_path = self.semantic_directory / "manifest.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if (
            manifest["model"] != self.MODEL_NAME
            or manifest["chunks_sha256"] !=
            self._chunk_fingerprint()
        ):
            raise ValueError(
                "Semantic index is outdated. Rebuild it."
            )

        self.chunks = self._load_chunks()

        self.embeddings = np.load(
            self.semantic_directory / "embeddings.npy",
            allow_pickle=False,
        )

        if (
            self.embeddings.ndim != 2
            or len(self.embeddings) != len(self.chunks)
        ):
            raise ValueError("Invalid semantic index dimensions")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Return sources ranked by cosine similarity."""
        if not query.strip() or k <= 0:
            return []

        if self.embeddings is None:
            self.load_index()

        assert self.embeddings is not None

        query_embedding = np.asarray(
            self.model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(-scores, kind="stable")[:k]

        sources: list[MinimalSource] = []

        for index in top_indices:
            chunk = self.chunks[int(index)]
            metadata = chunk["metadata"]
            start = int(metadata["start_index"])

            sources.append(
                MinimalSource(
                    file_path=str(metadata["file_path"]),
                    first_character_index=start,
                    last_character_index=(
                        start + len(chunk["page_content"])
                    ),
                )
            )

        return sources
