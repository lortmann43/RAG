
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .models.minimal_search_answer_models import MinimalSearchResults
from .models.student_search_answer_model import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .pipeline import RagPipeline
from .retriever import Retriever

from uuid import uuid4


INDEX_DIRECTORY = Path("data/processed")
SOURCE_DIRECTORY = Path("data/raw/vllm-0.10.1")

app = FastAPI(
    title="RAG Against the Machine",
    description="Local API for retrieving and answering vLLM questions.",
    version="1.0.0",
)


class AnswerRequest(BaseModel):
    """Input model for answer generation."""

    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=0, le=50)
    context_k: int = Field(default=3, ge=0, le=50)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Load the BM25 retriever once."""
    return Retriever(
        index_directory=INDEX_DIRECTORY,
        source_directory=SOURCE_DIRECTORY,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Load and reuse the RAG pipeline and Qwen model."""
    return RagPipeline(
        index_directory=INDEX_DIRECTORY,
        source_directory=SOURCE_DIRECTORY,
    )


@app.get("/search", response_model=StudentSearchResults)
def search(
    query: str = Query(min_length=1),
    k: int = Query(default=5, ge=0, le=50),
) -> StudentSearchResults:
    """Retrieve the top-k sources for a question."""
    if not query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty",
        )

    try:
        retriever = get_retriever()
        sources = retriever.search(query, k)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval unavailable: {exc}",
        ) from exc

    result = MinimalSearchResults(
        question_id=str(uuid4()),
        question=query,
        retrieved_sources=sources,
    )

    return StudentSearchResults(
        search_results=[result],
        k=k,
    )


@app.post(
    "/answer",
    response_model=StudentSearchResultsAndAnswer,
)
def answer(
    request: AnswerRequest,
) -> StudentSearchResultsAndAnswer:
    """Generate an answer with Qwen using retrieved sources."""
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    try:
        pipeline = get_pipeline()
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"RAG pipeline unavailable: {exc}",
        ) from exc

    try:
        result = pipeline.answer(
            question=request.question,
            k=request.k,
            context_k=request.context_k,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return StudentSearchResultsAndAnswer(
        search_results=[result],
        k=request.k,
    )
