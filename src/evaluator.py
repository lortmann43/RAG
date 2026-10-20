from .models.minimal_source_model import MinimalSource
from .models.question_models import AnsweredQuestion
from .models.rag_dataset_model import RagDataset
from .models.student_search_answer_model import StudentSearchResults


class Evaluator:
    """Calculate retrieval recall against ground truth-sources"""

    def __init__(self, dataset: RagDataset) -> None:
        self.questions = {
            question.question_id: question
            for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion)
        }

        if not self.questions:
            raise ValueError("No answered questions in dataset")

    @staticmethod
    def overlaps(
        source: MinimalSource,
        reference: MinimalSource
    ) -> bool:
        """Check whethet two source ranges have IoU of >= 0.05"""
        if source.file_path != reference.file_path:
            return False

        start = max(
            source.first_character_index,
            reference.first_character_index
        )

        end = min(
            source.last_character_index,
            reference.last_character_index
        )

        intersection = max(0, end - start)

        union = (
            max(
                source.last_character_index,
                reference.last_character_index
            )
            - min(
                source.first_character_index,
                reference.first_character_index
            )
        )

        if union <= 0:
            return False

        return intersection / union >= 0.05

    def recall_at_k(
            self,
            results: StudentSearchResults,
            k: int
    ) -> float:
        """Calculate mean recall@k across all questions"""
        if k < 0:
            raise ValueError("k cannot be negative")

        results_by_id = {
            result.question_id: result
            for result in results.search_results
        }

        scores: list[float] = []

        for question_id, question in self.questions.items():
            if not question.sources:
                continue

            result = results_by_id.get(question_id)

            retrieved = (
                result.retrieved_sources[:k]
                if result is not None else []
            )

            found = sum(
                any(
                    self.overlaps(source, reference)
                    for source in retrieved
                )
                for reference in question.sources
            )

            scores.append(found / len(question.sources))

        return sum(scores) / len(scores) if scores else 0.0
