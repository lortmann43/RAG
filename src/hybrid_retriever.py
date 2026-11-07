
from pathlib import Path

from .models.minimal_source_model import MinimalSource
from .retriever import Retriever
from .semantic_retriever import SemanticRetriever


def reciprocal_rank_fusion(
    bm25_results: list[MinimalSource],
    semantic_results: list[MinimalSource],
    k: int = 5,
    bm25_weight: float = 2.0,
    semantic_weight: float = 1.0,
    rank_constant: int = 60,
) -> list[MinimalSource]:
    """Combine two ranked result lists using weighted RRF."""
    if k <= 0:
        return []

    if rank_constant < 0:
        raise ValueError("rank_constant cannot be negative")

    if bm25_weight <= 0 or semantic_weight <= 0:
        raise ValueError("Retrieval weights must be positive")

    scores: dict[tuple[str, int, int], float] = {}
    sources: dict[tuple[str, int, int], MinimalSource] = {}

    rankings = (
        (bm25_results, bm25_weight),
        (semantic_results, semantic_weight),
    )

    for results, weight in rankings:
        for rank, source in enumerate(results, start=1):
            key = (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )

            sources[key] = source
            scores[key] = scores.get(key, 0.0) + (
                weight / (rank_constant + rank)
            )

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [sources[key] for key in ranked_keys[:k]]


class HybridRetriever:
    """Combine BM25 and semantic retrieval."""

    def __init__(
        self,
        index_directory: Path,
        source_directory: Path,
    ) -> None:
        self.bm25 = Retriever(
            index_directory=index_directory,
            source_directory=source_directory,
        )

        self.semantic = SemanticRetriever(index_directory)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Retrieve and fuse BM25 and semantic results."""
        if not query.strip() or k <= 0:
            return []

        # Retrieve more candidates than the final result count.
        candidate_k = max(20, k * 4)

        bm25_results = self.bm25.search(query, candidate_k)
        semantic_results = self.semantic.search(
            query, candidate_k
        )

        return reciprocal_rank_fusion(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
            k=k,
        )
