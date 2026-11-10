
import hashlib
import json
from pathlib import Path
from typing import Protocol

from .models.minimal_source_model import MinimalSource


class SearchBackend(Protocol):
    """Interface shared by retrieval implementations."""

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Retrieve source locations."""
        ...


class CachedRetriever:
    """Cache retrieval results persistently on disk."""

    CACHE_VERSION = "1"

    def __init__(
        self,
        backend: SearchBackend,
        index_directory: Path,
        method: str,
    ) -> None:
        self.backend = backend
        self.method = method
        self.cache_directory = index_directory / "query_cache"

        self.cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks_path = index_directory / "chunks.jsonl"

        self.index_hash = hashlib.sha256(
            chunks_path.read_bytes()
        ).hexdigest()

        self.hits = 0
        self.misses = 0

    def _cache_path(self, query: str, k: int) -> Path:
        """Build a unique key for this retrieval request."""
        key_data = {
            "version": self.CACHE_VERSION,
            "index": self.index_hash,
            "method": self.method,
            "query": query,
            "k": k,
        }

        encoded = json.dumps(
            key_data,
            sort_keys=True,
        ).encode("utf-8")

        key = hashlib.sha256(encoded).hexdigest()

        return self.cache_directory / f"{key}.json"

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[MinimalSource]:
        """Return cached results or execute retrieval."""
        if not query.strip() or k <= 0:
            return self.backend.search(query, k)

        path = self._cache_path(query, k)

        if path.exists():
            try:
                data = json.loads(
                    path.read_text(encoding="utf-8")
                )

                sources = [
                    MinimalSource.model_validate(item)
                    for item in data
                ]

                self.hits += 1
                return sources

            except (OSError, ValueError):
                # Ignore corrupted cache files.
                pass

        self.misses += 1

        sources = self.backend.search(query, k)

        data = [
            source.model_dump()
            for source in sources
        ]

        temp_path = path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        temp_path.replace(path)

        return sources
