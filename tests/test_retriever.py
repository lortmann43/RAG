
from pathlib import Path

import pytest

from src.indexer import Indexer
from src.retriever import Retriever


@pytest.fixture
def small_index(tmp_path: Path) -> tuple[Retriever, Path]:
    """Create a small BM25 index for testing."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    files = {
        "astronomy.md": (
            "# Astronomy\n"
            "A quasar is an extremely luminous galactic nucleus.\n"
        ),
        "cooking.txt": (
            "Risotto is made using rice, broth, and vegetables.\n"
        ),
        "math.py": (
            "def calculate_factorial(n: int) -> int:\n"
            "    return 1 if n <= 1 else n * calculate_factorial(n - 1)\n"
        ),
    }

    for filename, content in files.items():
        (corpus / filename).write_text(
            content,
            encoding="utf-8",
        )

    index_directory = tmp_path / "index"

    indexer = Indexer(source_directory=corpus)
    indexer.build_index()
    indexer.save_index(index_directory)

    retriever = Retriever(
        index_directory=index_directory,
        source_directory=corpus,
    )

    return retriever, corpus


def test_retrieves_correct_file(
    small_index: tuple[Retriever, Path],
) -> None:
    """Find the document containing the query term."""
    retriever, corpus = small_index

    sources = retriever.search("quasar", k=3)

    assert sources
    assert sources[0].file_path == str(corpus / "astronomy.md")


def test_retrieved_offsets(
    small_index: tuple[Retriever, Path],
) -> None:
    """Verify that retrieved offsets point to the source text."""
    retriever, _ = small_index

    sources = retriever.search("calculate_factorial", k=3)

    assert sources

    source = sources[0]
    text = Path(source.file_path).read_text(encoding="utf-8")

    snippet = text[
        source.first_character_index:source.last_character_index
    ]

    assert "calculate_factorial" in snippet


def test_k_limit(
    small_index: tuple[Retriever, Path],
) -> None:
    """Never return more than k results."""
    retriever, _ = small_index

    sources = retriever.search("quasar", k=1)

    assert len(sources) <= 1


def test_empty_query(
    small_index: tuple[Retriever, Path],
) -> None:
    """An empty query should return no results."""
    retriever, _ = small_index

    assert retriever.search("", k=5) == []
    assert retriever.search("quasar", k=0) == []
