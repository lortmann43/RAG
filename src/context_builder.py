from pathlib import Path
from .models.minimal_source_model import MinimalSource


def build_context(sources: list[MinimalSource]) -> str:
    """Build a context string from retrieved source locations."""
    context_parts: list[str] = []

    for i, source in enumerate(sources, start=1):
        path = Path(source.file_path)

        with path.open("r", encoding="utf-8", newline="") as file:
            text = file.read()

        start = source.first_character_index
        end = source.last_character_index

        if not 0 <= start < end <= len(text):
            raise ValueError(f"Invalid character range in {path}")

        snippet = text[start:end]

        context_parts.append(
            f"Source {i}: {source.file_path}\n"
            f"Characters: {start} - {end}\n"
            f"{snippet}"
        )

    return "\n\n".join(context_parts)