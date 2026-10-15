import fire
from pathlib import Path
from time import perf_counter
from .indexer import Indexer
from tqdm import tqdm
from .retriever import Retriever
from .models.rag_dataset_model import RagDataset
from .models.minimal_search_answer_models import MinimalSearchResults
from .models.student_search_answer_model import StudentSearchResults


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


if __name__ == "__main__":
    fire.Fire(RagCLI)
