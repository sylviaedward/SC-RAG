from __future__ import annotations

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


class SearchResponse(BaseModel):
    query: str
    expanded_query: str
    hits: list[SearchHit]

