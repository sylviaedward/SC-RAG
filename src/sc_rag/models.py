from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    alias: str
    title: str
    page: int = Field(ge=1)
    text: str


class SearchHit(Chunk):
    score: float
    rank: int = Field(ge=1)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    expanded_query: str
    hits: list[SearchHit]


class AnswerMode(str, Enum):
    auto = "auto"
    fundamentals = "fundamentals"
    paper_lookup = "paper_lookup"
    comparison = "comparison"


class EvidenceSufficiency(str, Enum):
    sufficient = "sufficient"
    insufficient = "insufficient"
    conflicting = "conflicting"


class AnswerStatus(str, Enum):
    answered = "answered"
    abstained = "abstained"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=2)
    mode: AnswerMode = AnswerMode.auto


class Citation(BaseModel):
    evidence_id: str
    chunk_id: str
    paper_id: str
    alias: str
    title: str
    page: int = Field(ge=1)
    excerpt: str


class AnswerClaim(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class AnswerResponse(BaseModel):
    question: str
    mode: AnswerMode
    status: AnswerStatus
    answer: str
    claims: list[AnswerClaim]
    evidence_sufficiency: EvidenceSufficiency
    confidence: float = Field(ge=0, le=1)
    reason: str
    retrieved_chunks: int = Field(ge=0)
    model: str

