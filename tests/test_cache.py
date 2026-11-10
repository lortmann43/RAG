
from pathlib import Path

from src.cached_retriever import CachedRetriever
from src.models.minimal_source_model import MinimalSource


class FakeBackend:
    """A fake retriever that counts search calls."""

    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        self.calls += 1

        return [
            MinimalSource(
                file_path="example.md",
                first_character_index=0,
                last_character_index=10,
            )
        ]


def test_cache_skips_retrieval(tmp_path: Path) -> None:
    """Repeated queries must not call the backend again."""
    (tmp_path / "chunks.jsonl").write_text(
        '{"page_content": "example"}\n',
        encoding="utf-8",
    )

    backend = FakeBackend()

    retriever = CachedRetriever(
        backend=backend,
        index_directory=tmp_path,
        method="bm25",
    )

    first = retriever.search("test query", k=5)
    second = retriever.search("test query", k=5)

    assert first == second
    assert backend.calls == 1
    assert retriever.hits == 1
    assert retriever.misses == 1

    # Verify that the cache survives a new instance.
    new_retriever = CachedRetriever(
        backend=backend,
        index_directory=tmp_path,
        method="bm25",
    )

    third = new_retriever.search("test query", k=5)

    assert third == first
    assert backend.calls == 1
    assert new_retriever.hits == 1
