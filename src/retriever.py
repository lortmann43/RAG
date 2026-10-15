from pathlib import Path
import bm25s
from .indexer import Indexer
from .models.minimal_source_model import MinimalSource


class Retriever:
    """Retrieve relevant source locations using BM25."""

    def __init__(
            self,
            index_directory: Path,
            source_directory: Path,
    ) -> None:
        self.indexer = Indexer(source_directory)
        self.indexer.load_index(index_directory)

    def search(
            self,
            query: str,
            k: int = 5,
    ) -> list[MinimalSource]:
        """Return the top-k relevant source locations."""
        if not query.strip() or k <= 0:
            return []

        bm25 = self.indexer.bm25
        if bm25 is None:
            raise ValueError("BM25 index is not loaded")

        chunks = self.indexer.chunks
        if not chunks:
            return []

        query_tokens = bm25s.tokenize(query)
        results, scores = bm25.retrieve(
            query_tokens,
            k=min(k, len(chunks))
        )

        sources: list[MinimalSource] = []

        for doc_id, score in zip(results[0], scores[0]):
            if float(score) <= 0:
                continue

            chunk = chunks[int(doc_id)]
            start = int(chunk.metadata["start_index"])

            sources.append(
                MinimalSource(
                    file_path=str(chunk.metadata["file_path"]),
                    first_character_index=start,
                    last_character_index=start + len(chunk.page_content)
                )
            )

        return sources
