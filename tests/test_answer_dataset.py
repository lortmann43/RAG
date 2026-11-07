
from pathlib import Path
from unittest.mock import patch

import pytest

from src.__main__ import RagCLI
from src.models.minimal_search_answer_models import (
    MinimalAnswer,
    MinimalSearchResults,
)
from src.models.student_search_answer_model import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def test_checkpoint_and_resume(tmp_path: Path) -> None:
    """Resume after interruption without repeating completed work."""

    questions = [
        MinimalSearchResults(
            question_id="q1",
            question="First question",
            retrieved_sources=[],
        ),
        MinimalSearchResults(
            question_id="q2",
            question="Second question",
            retrieved_sources=[],
        ),
    ]

    dataset = StudentSearchResults(
        search_results=questions,
        k=5,
    )

    input_path = tmp_path / "questions.json"
    input_path.write_text(
        dataset.model_dump_json(indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "answers"
    output_path = output_dir / "questions.json"

    def fake_answer(question: MinimalSearchResults) -> MinimalAnswer:
        return MinimalAnswer(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=question.retrieved_sources,
            answer=f"Mock answer for {question.question_id}",
        )

    with patch("src.__main__.RagPipeline") as mock_pipeline:
        generate = mock_pipeline.return_value.answer_from_sources

        # First execution: succeed once, then simulate a crash.
        generate.side_effect = [
            fake_answer(questions[0]),
            RuntimeError("Simulated interruption"),
        ]

        with pytest.raises(RuntimeError, match="Simulated interruption"):
            RagCLI().answer_dataset(
                student_search_results_path=str(input_path),
                save_directory=str(output_dir),
            )

        # The first answer should already be saved.
        checkpoint = StudentSearchResultsAndAnswer.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

        assert len(checkpoint.search_results) == 1
        assert checkpoint.search_results[0].question_id == "q1"

        # Second execution: resume and generate only q2.
        generate.reset_mock()
        generate.side_effect = [fake_answer(questions[1])]

        RagCLI().answer_dataset(
            student_search_results_path=str(input_path),
            save_directory=str(output_dir),
        )

        assert generate.call_count == 1
        assert generate.call_args.kwargs["question_id"] == "q2"

        completed = StudentSearchResultsAndAnswer.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

        assert len(completed.search_results) == 2
        assert [
            answer.question_id for answer in completed.search_results
        ] == ["q1", "q2"]

        # Third execution: all answers exist, so don't load Qwen.
        mock_pipeline.reset_mock()

        RagCLI().answer_dataset(
            student_search_results_path=str(input_path),
            save_directory=str(output_dir),
        )

        mock_pipeline.assert_not_called()
