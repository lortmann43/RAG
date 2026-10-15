import fire
from pathlib import Path
from time import perf_counter
from .indexer import Indexer


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


if __name__ == "__main__":
    fire.Fire(RagCLI)
