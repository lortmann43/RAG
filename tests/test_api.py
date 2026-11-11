
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.models.minimal_source_model import MinimalSource
from src.models.minimal_search_answer_models import MinimalAnswer


def test_search_endpoint() -> None:
    """Search endpoint returns valid source locations."""
    source = MinimalSource(
        file_path="docs/example.md",
        first_character_index=0,
        last_character_index=100,
    )

    with patch("src.api.get_retriever") as get_retriever:
        retriever = get_retriever.return_value
        retriever.search.return_value = [source]

        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={"query": "What is vLLM?", "k": 5},
            )

        assert response.status_code == 200

        data = response.json()
        assert data["k"] == 5
        assert len(data["search_results"]) == 1

        result = data["search_results"][0]
        assert result["question"] == "What is vLLM?"
        assert result["retrieved_sources"] == [
            source.model_dump()
        ]

        retriever.search.assert_called_once_with(
            "What is vLLM?", 5
        )


def test_answer_endpoint() -> None:
    """Answer endpoint uses the existing RAG pipeline."""
    answer = MinimalAnswer(
        question_id="test-123",
        question="What is vLLM?",
        retrieved_sources=[],
        answer="vLLM is an LLM inference engine.",
    )

    with patch("src.api.get_pipeline") as get_pipeline:
        pipeline = get_pipeline.return_value
        pipeline.answer.return_value = answer

        with TestClient(app) as client:
            response = client.post(
                "/answer",
                json={
                    "question": "What is vLLM?",
                    "k": 5,
                    "context_k": 3,
                },
            )

        assert response.status_code == 200

        data = response.json()
        assert data["k"] == 5
        assert data["search_results"][0]["answer"] == (
            "vLLM is an LLM inference engine."
        )

        pipeline.answer.assert_called_once_with(
            question="What is vLLM?",
            k=5,
            context_k=3,
        )


def test_invalid_search_k() -> None:
    """Negative k is rejected by FastAPI validation."""
    with TestClient(app) as client:
        response = client.get(
            "/search",
            params={"query": "test", "k": -1},
        )

    assert response.status_code == 422


def test_empty_answer_question() -> None:
    """Empty question is rejected without loading Qwen."""
    with patch("src.api.get_pipeline") as get_pipeline:
        with TestClient(app) as client:
            response = client.post(
                "/answer",
                json={"question": "", "k": 5},
            )

        assert response.status_code == 422
        get_pipeline.assert_not_called()
