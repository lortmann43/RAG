
from pathlib import Path
from unittest.mock import patch

import pytest

from src.__main__ import RagCLI


@pytest.mark.parametrize("content", [None, "{invalid-json"])
def test_invalid_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str | None,
) -> None:
    """Missing and malformed datasets must fail gracefully."""
    dataset_path = tmp_path / "dataset.json"

    if content is not None:
        dataset_path.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        RagCLI().search_dataset(
            dataset_path=str(dataset_path),
        )

    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "Dataset search failed:" in captured.err


def test_missing_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing BM25 index must produce a controlled error."""
    with pytest.raises(SystemExit) as exc:
        RagCLI().search(
            query="What is vLLM?",
            index_directory=str(tmp_path / "missing_index"),
        )

    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "Search failed:" in captured.err


def test_empty_answer_question(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty question must fail without loading Qwen."""
    with patch("src.__main__.RagPipeline") as mock_pipeline:
        with pytest.raises(SystemExit) as exc:
            RagCLI().answer(query="")

        assert exc.value.code == 1
        mock_pipeline.assert_not_called()

    captured = capsys.readouterr()
    assert "Question cannot be empty" in captured.err
