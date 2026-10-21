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
    partial = "partial"
    insufficient = "insufficient"
    conflicting = "conflicting"


class AnswerStatus(str, Enum):
    answered = "answered"
    partial = "partial"
    abstained = "abstained"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=2)
    mode: AnswerMode = AnswerMode.auto


class Citation(BaseModel):
    evidence_id: str
    span_id: str
    chunk_id: str
    paper_id: str
    alias: str
    title: str
    page: int = Field(ge=1)
    excerpt: str


class AnswerClaim(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class AgentStep(BaseModel):
    action: str
    status: str
    detail: str = ""


class AgentTrace(BaseModel):
    task_type: AnswerMode
    plan: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    retrieval_rounds: int = Field(default=0, ge=0)
    stop_reason: str = ""


class AnswerResponse(BaseModel):
    question: str
    mode: AnswerMode
    status: AnswerStatus
    answer: str
    claims: list[AnswerClaim]
    evidence_sufficiency: EvidenceSufficiency
    confidence: float = Field(ge=0, le=1)
    reason: str
    missing_information: list[str] = Field(default_factory=list)
    retrieved_chunks: int = Field(ge=0)
    candidate_chunks: int = Field(default=0, ge=0)
    evidence_aliases: list[str] = Field(default_factory=list)
    requested_aliases: list[str] = Field(default_factory=list)
    missing_requested_aliases: list[str] = Field(default_factory=list)
    generation_attempts: int = Field(default=1, ge=0)
    repair_reason: str | None = None
    model: str
    agent_trace: AgentTrace | None = None


class ReviewDimension(str, Enum):
    method = "method"
    bitstream = "bitstream"
    hardware = "hardware"
    evaluation = "evaluation"
    limitations = "limitations"


class LiteratureReviewRequest(BaseModel):
    topic: str = Field(min_length=3)
    aliases: list[str] = Field(min_length=2, max_length=8)
    chunks_per_paper: int = Field(default=5, ge=1, le=8)


class ReviewFact(BaseModel):
    fact_id: str
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class PaperReviewRecord(BaseModel):
    alias: str
    paper_id: str
    title: str
    method: list[ReviewFact] = Field(default_factory=list)
    bitstream: list[ReviewFact] = Field(default_factory=list)
    hardware: list[ReviewFact] = Field(default_factory=list)
    evaluation: list[ReviewFact] = Field(default_factory=list)
    limitations: list[ReviewFact] = Field(default_factory=list)
    missing_dimensions: list[ReviewDimension] = Field(default_factory=list)


class ComparisonCell(BaseModel):
    alias: str
    summary: str
    fact_ids: list[str] = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class EvidenceConflict(BaseModel):
    description: str
    fact_ids: list[str] = Field(min_length=2)
    citations: list[Citation] = Field(min_length=2)


class ComparisonRow(BaseModel):
    dimension: ReviewDimension
    cells: list[ComparisonCell]
    conflicts: list[EvidenceConflict] = Field(default_factory=list)


class ResearchIdea(BaseModel):
    observed_gap: str
    evidence_fact_ids: list[str] = Field(min_length=1)
    evidence: list[Citation] = Field(min_length=1)
    inferred_limitation: str
    hypothesis: str
    proposed_experiment: str
    falsifier: str
    uncertainty: str
    novelty_status: str = "unverified"


class LiteratureReviewResponse(BaseModel):
    topic: str
    requested_aliases: list[str]
    missing_aliases: list[str] = Field(default_factory=list)
    papers: list[PaperReviewRecord]
    comparison: list[ComparisonRow]
    ideas: list[ResearchIdea]
    model: str
    prompt_version: str

