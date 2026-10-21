from pathlib import Path
from uuid import uuid4
from .context_builder import build_context
from .generator import Generator
from .retriever import Retriever
from .models.minimal_source_model import MinimalSource
from .models.minimal_search_answer_models import MinimalAnswer


class RagPipeline:
    """Combine retrieval, context building, and answer generation."""

    def __init__(
            self,
            index_directory: Path,
            source_directory: Path
    ) -> None:
        self.retriever = Retriever(
            index_directory=index_directory,
            source_directory=source_directory
        )
        self.generator = Generator()

    def answer_from_sources(
            self,
            question: str,
            sources: list[MinimalSource],
            question_id: str | None = None
    ) -> MinimalAnswer:
        """Generate an answer from previously retrieved sources."""
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if sources:
            context = build_context(sources)
            answer = self.generator.generate(question, context)
        else:
            answer = "Not enough information to answer"

        return MinimalAnswer(
            question_id=question_id or str(uuid4()),
            question=question,
            retrieved_sources=sources,
            answer=answer
        )

    def answer(
            self,
            question: str,
            k: int = 5
    ) -> MinimalAnswer:
        """Retrieve relevant sources and generate an answer."""
        if k < 0:
            raise ValueError("k cannot be negative")

        sources = self.retriever.search(question, k)

        return self.answer_from_sources(question, sources)
