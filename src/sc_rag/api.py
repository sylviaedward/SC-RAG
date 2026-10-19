from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from .answering import GroundedAnswerService, OpenAICompatibleChatClient, build_hybrid_retriever, load_known_aliases
from .literature_review import LiteratureReviewService
from .models import AnswerRequest, AnswerResponse, LiteratureReviewRequest, LiteratureReviewResponse, SearchResponse
from .search import BM25Index


app = FastAPI(title="SC-RAG", version="0.1.0")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browser users to the interactive API documentation."""
    return RedirectResponse(url="/docs")


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    path = Path(os.getenv("SC_RAG_INDEX", "artifacts/chunks.jsonl"))
    if not path.exists():
        raise FileNotFoundError(f"Index is missing: {path}. Run `sc-rag-ingest` first.")
    return BM25Index.from_jsonl(path)


@lru_cache(maxsize=1)
def get_answer_service() -> GroundedAnswerService:
    chunks = Path(os.getenv("SC_RAG_INDEX", "artifacts/chunks.jsonl"))
    return GroundedAnswerService(
        build_hybrid_retriever(
            chunks=chunks,
            matrix=Path(os.getenv("SC_RAG_DENSE_MATRIX", "artifacts/dense/embeddings.npy")),
            metadata=Path(os.getenv("SC_RAG_DENSE_METADATA", "artifacts/dense/metadata.json")),
            embedding_provider=os.getenv("SC_RAG_EMBEDDING_PROVIDER", "local"),
            embedding_model=os.getenv("SC_RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        ),
        OpenAICompatibleChatClient.from_env(),
        known_aliases=load_known_aliases(chunks),
    )


@lru_cache(maxsize=1)
def get_literature_review_service() -> LiteratureReviewService:
    return LiteratureReviewService(get_index(), OpenAICompatibleChatClient.from_env())


@app.get("/health")
def health() -> dict[str, object]:
    try:
        index = get_index()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok", "chunks": len(index.chunks), "retriever": "bm25"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=2),
    limit: int = Query(default=8, ge=1, le=20),
) -> SearchResponse:
    try:
        return get_index().search(q, limit)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    try:
        return get_answer_service().answer(request.question, request.mode)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/literature-review", response_model=LiteratureReviewResponse)
def literature_review(request: LiteratureReviewRequest) -> LiteratureReviewResponse:
    try:
        return get_literature_review_service().review(
            request.topic,
            request.aliases,
            request.chunks_per_paper,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

