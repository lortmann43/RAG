
import pytest

from src.evaluator import Evaluator
from src.models.minimal_source_model import MinimalSource
from src.models.question_models import AnsweredQuestion
from src.models.rag_dataset_model import RagDataset
from src.models.minimal_search_answer_models import MinimalSearchResults
from src.models.student_search_answer_model import StudentSearchResults


def source(path: str, start: int, end: int) -> MinimalSource:
    """Create a source location for testing."""
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def test_overlapping_sources() -> None:
    """Overlapping ranges in the same file should match."""
    reference = source("example.py", 10, 110)
    retrieved = source("example.py", 90, 130)

    assert Evaluator.overlaps(retrieved, reference)


def test_different_files() -> None:
    """Identical ranges in different files should not match."""
    reference = source("example.py", 10, 110)
    retrieved = source("other.py", 10, 110)

    assert not Evaluator.overlaps(retrieved, reference)


def test_non_overlapping_sources() -> None:
    """Separate character ranges should not match."""
    reference = source("example.py", 10, 110)
    retrieved = source("example.py", 150, 250)

    assert not Evaluator.overlaps(retrieved, reference)


def test_insufficient_overlap() -> None:
    """An IoU below 0.05 should not count as a match."""
    reference = source("example.py", 0, 100)
    retrieved = source("example.py", 99, 199)

    assert not Evaluator.overlaps(retrieved, reference)


def test_recall_at_k() -> None:
    """Recall should increase when relevant sources are retrieved."""
    question = AnsweredQuestion(
        question_id="test-1",
        question="Example question",
        sources=[
            source("first.py", 0, 100),
            source("second.py", 0, 100),
        ],
        answer="Example answer",
    )

    dataset = RagDataset(rag_questions=[question])

    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="test-1",
                question="Example question",
                retrieved_sources=[
                    source("first.py", 10, 90),
                    source("second.py", 10, 90),
                ],
            )
        ],
        k=2,
    )

    evaluator = Evaluator(dataset)

    assert evaluator.recall_at_k(results, 1) == pytest.approx(0.5)
    assert evaluator.recall_at_k(results, 2) == pytest.approx(1.0)
