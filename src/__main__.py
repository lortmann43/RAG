import fire
from pathlib import Path
from time import perf_counter
from .indexer import Indexer
from tqdm import tqdm
from .retriever import Retriever
from .evaluator import Evaluator
from .pipeline import RagPipeline
from .models.rag_dataset_model import RagDataset
from .models.minimal_search_answer_models import (MinimalSearchResults,
                                                  MinimalAnswer)
from .models.student_search_answer_model import (
    StudentSearchResults, StudentSearchResultsAndAnswer)
from uuid import uuid4


class RagCLI:
    """Command-line interface for the RAG pipeline."""

    def index(
            self,
            max_chunk_size: int = 2000,
            source_directory: str = "data/raw/vllm-0.10.1",
            save_directory: str = "data/processed"
    ) -> None:
        """Build and save the searchable BM25 index"""
        start_time = perf_counter()

        try:
            indexer = Indexer(
                source_directory=Path(source_directory),
                max_chunk_size=max_chunk_size)

            indexer.build_index()
            indexer.save_index(Path(save_directory))

        except (OSError, ValueError) as exc:
            print(f"Indexing failed: {exc}")
            return

        elapsed = perf_counter() - start_time
        print(f"Indexed {len(indexer.chunks)} chunks")
        print(f"Index saved to: {save_directory}")
        print(f"Indexing time: {elapsed:.2f} seconds")

    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = "data/processed",
        source_directory: str = "data/raw/vllm-0.10.1",
    ) -> None:
        """Search the index for relevant source locations."""
        try:
            if k < 0:
                raise ValueError("k cannot be negative")

            retriever = Retriever(
                index_directory=Path(index_directory),
                source_directory=Path(source_directory),
            )

            sources = retriever.search(query, k)

            result = MinimalSearchResults(
                question_id=str(uuid4()),
                question=query,
                retrieved_sources=sources,
            )

            output = StudentSearchResults(
                search_results=[result],
                k=k,
            )

            print(output.model_dump_json(indent=2))

        except (OSError, ValueError) as exc:
            print(f"Search failed: {exc}")

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 5,
            save_directory: str = "data/output/search_results",
            index_directory: str = "data/processed",
            source_directory: str = "data/raw/vllm-0.10.1",
    ) -> None:
        """Retrieve sources for all questions in a dataset."""
        try:
            if k < 0:
                raise ValueError("k cannot be negative")

            path = Path(dataset_path)
            dataset = RagDataset.model_validate_json(
                path.read_text(encoding="utf-8")
            )

            retriever = Retriever(
                index_directory=Path(index_directory),
                source_directory=Path(source_directory)
            )

            results: list[MinimalSearchResults] = []

            for question in tqdm(
                dataset.rag_questions,
                desc="Searching",
                unit="question"
            ):
                sources = retriever.search(question.question, k)

                results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources
                    )
                )

            output = StudentSearchResults(
                search_results=results,
                k=k
            )

            directory = Path(save_directory)
            directory.mkdir(parents=True, exist_ok=True)

            output_path = directory / path.name
            output_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )

            print(f"Saved {len(results)} results to {output_path}")

        except (OSError, ValueError) as exc:
            print(f"Dataset search failed: {exc}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality against ground truth."""
        try:
            results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(
                    encoding="utf-8"
                )
            )

            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(
                    encoding="utf-8"
                )
            )

            evaluator = Evaluator(dataset)

            print("\nEvaluation Results")
            print("=" * 40)

            for k in (1, 3, 5, 10):
                if k > results.k:
                    continue

                score = evaluator.recall_at_k(results, k)
                print(f"Recall@{k}: {score:.2%}")

        except (OSError, ValueError) as exc:
            print(f"Evaluation failed: {exc}")

    def answer(
            self,
            query: str,
            k: int = 5,
            index_directory: str = "data/processed",
            source_directory: str = "data/raw/vllm-0.10.1",
            save_directory: str =
            "data/output/search_results_and_answer/single",
            context_k: int = 3
    ) -> None:
        """Retrieve sources and generate an answer using Qwen."""
        try:
            if not query.strip():
                raise ValueError("Question cannot be empty")

            if k < 0:
                raise ValueError("k cannot be negative")

            pipeline = RagPipeline(
                index_directory=Path(index_directory),
                source_directory=Path(source_directory)
            )

            result = pipeline.answer(query, k, context_k=context_k)

            output = StudentSearchResultsAndAnswer(
                search_results=[result],
                k=k
            )

            directory = Path(save_directory)
            directory.mkdir(parents=True, exist_ok=True)

            output_path = directory / f"{result.question_id}.json"
            output_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )

            print(output.model_dump_json(indent=2))
            print(f"\nSaved answer to: {output_path}")

        except (OSError, ValueError) as exc:
            print(f"Answer generation failed: {exc}")

    def answer_dataset(
        self,
        search_results_path: str,
        context_k: int = 3,
        save_directory: str = "data/output/search_results_and_answer",
        index_directory: str = "data/processed",
        source_directory: str = "data/raw/vllm-0.10.1",
        resume: bool = True,
    ) -> None:
        """Generate answers and resume from existing checkpoints."""
        try:
            if context_k < 0:
                raise ValueError("context_k cannot be negative")

            path = Path(search_results_path)
            search_results = StudentSearchResults.model_validate_json(
                path.read_text(encoding="utf-8")
            )

            directory = Path(save_directory)
            directory.mkdir(parents=True, exist_ok=True)
            output_path = directory / path.name

            completed: dict[str, MinimalAnswer] = {}

            # Load previously generated answers
            if resume and output_path.exists():
                previous = StudentSearchResultsAndAnswer.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                )

                if previous.k != search_results.k:
                    raise ValueError("Checkpoint k does not match input")

                original_by_id = {
                    item.question_id: item
                    for item in search_results.search_results
                }

                for answer in previous.search_results:
                    original = original_by_id.get(answer.question_id)

                    if (
                        original is None
                        or answer.question != original.question
                        or answer.retrieved_sources !=
                        original.retrieved_sources
                    ):
                        raise ValueError("Checkpoint does not match input")

                    completed[answer.question_id] = answer

                print(f"Resuming: {len(completed)} answers already saved")

            pipeline: RagPipeline | None = None

            for result in tqdm(
                search_results.search_results,
                desc="Generating answers",
                unit="question",
            ):
                if result.question_id in completed:
                    continue

                # Only load Qwen when generation is needed
                if pipeline is None:
                    pipeline = RagPipeline(
                        index_directory=Path(index_directory),
                        source_directory=Path(source_directory),
                    )

                answer = pipeline.answer_from_sources(
                    question=result.question,
                    sources=result.retrieved_sources,
                    question_id=result.question_id,
                    context_k=context_k,
                )

                completed[result.question_id] = answer

                # Preserve the original question order
                answers = [
                    completed[item.question_id]
                    for item in search_results.search_results
                    if item.question_id in completed
                ]

                output = StudentSearchResultsAndAnswer(
                    search_results=answers,
                    k=search_results.k,
                )

                temp_path = output_path.with_suffix(".json.tmp")
                temp_path.write_text(
                    output.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                temp_path.replace(output_path)

            print(
                f"Completed {len(completed)}/"
                f"{len(search_results.search_results)} questions"
            )

        except (OSError, ValueError) as exc:
            print(f"Dataset answer generation failed: {exc}")


if __name__ == "__main__":
    fire.Fire(RagCLI)
