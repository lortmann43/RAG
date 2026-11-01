
from pathlib import Path

import pytest

from src.chunkers import Chunker


@pytest.mark.parametrize(
    "filename, content",
    [
        (
            "example.py",
            "".join(
                f"def function_{i}():\n    return {i}\n\n"
                for i in range(40)
            ),
        ),
        (
            "example.md",
            "".join(
                f"## Section {i}\nSome documentation text.\n\n"
                for i in range(40)
            ),
        ),
        (
            "example.txt",
            "".join(
                f"Section {i}: Some plain text content.\n"
                for i in range(40)
            ),
        ),
    ],
)
def test_chunk_offsets(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    """Verify chunk sizes and original character positions."""
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")

    chunker = Chunker(max_chunk_size=120)
    chunks = chunker.split_file(path)

    assert len(chunks) > 1

    for chunk in chunks:
        start = chunk.metadata["start_index"]
        end = start + len(chunk.page_content)

        assert len(chunk.page_content) <= 120
        assert content[start:end] == chunk.page_content
        assert chunk.metadata["file_path"] == path.as_posix()


@pytest.mark.parametrize("chunk_size", [0, -1, 2001])
def test_invalid_chunk_size(chunk_size: int) -> None:
    """Reject chunk sizes outside the allowed range."""
    with pytest.raises(ValueError):
        Chunker(max_chunk_size=chunk_size)
