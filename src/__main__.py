import fire


class RagCLI:
    """Command-line interface for the RAG pipeline."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the searchable index for the codebase."""
        if not 1 <= max_chunk_size <= 2000:
            print("Error: chunk size must be between 1 and 2000")
            return

        print(f"Indexing with chunk size: {max_chunk_size}")


if __name__ == "__main__":
    fire.Fire(RagCLI)
