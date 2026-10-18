from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Protocol, Sequence

from .dense import DEFAULT_EMBEDDING_MODEL, DenseIndex, create_embedder
from .models import SearchHit, SearchResponse
from .search import BM25Index


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("Install Phase 1 dependencies with `make setup-hybrid`.") from error
        self.model_name = model_name
        import torch

        torch.set_num_threads(int(os.getenv("SC_RAG_TORCH_THREADS", "2")))
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        values = self.model.predict(
            [(query, passage) for passage in passages],
            batch_size=8,
            show_progress_bar=False,
        )
        return [float(value) for value in values]


def reciprocal_rank_fusion(
    bm25_hits: Sequence[SearchHit],
    dense_hits: Sequence[SearchHit],
    rrf_k: int = 60,
) -> list[SearchHit]:
    fused: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for source, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id not in fused:
                fused[hit.chunk_id] = hit.model_copy(deep=True)
                scores[hit.chunk_id] = 0.0
            scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
            if source == "bm25":
                fused[hit.chunk_id].bm25_rank = rank
            else:
                fused[hit.chunk_id].dense_rank = rank
    ordered = sorted(fused.values(), key=lambda hit: (-scores[hit.chunk_id], hit.chunk_id))
    for rank, hit in enumerate(ordered, start=1):
        hit.rank = rank
        hit.rrf_score = round(scores[hit.chunk_id], 8)
        hit.score = hit.rrf_score
    return ordered


def apply_paper_diversity(hits: Sequence[SearchHit], limit: int, max_per_paper: int) -> list[SearchHit]:
    selected: list[SearchHit] = []
    paper_counts: dict[str, int] = {}
    for hit in hits:
        if paper_counts.get(hit.paper_id, 0) >= max_per_paper:
            continue
        selected.append(hit.model_copy(deep=True))
        paper_counts[hit.paper_id] = paper_counts.get(hit.paper_id, 0) + 1
        if len(selected) == limit:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        reranker: Reranker | None = None,
        candidate_k: int = 40,
        rerank_k: int = 30,
        rrf_k: int = 60,
        max_per_paper: int = 2,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.max_per_paper = max_per_paper
        self.name = "hybrid_rrf_reranked" if reranker else "hybrid_rrf"

    def search(self, query: str, limit: int = 8) -> SearchResponse:
        bm25_response = self.bm25.search(query, self.candidate_k)
        dense_response = self.dense.search(query, self.candidate_k)
        fused = reciprocal_rank_fusion(bm25_response.hits, dense_response.hits, self.rrf_k)
        if self.reranker:
            candidates = fused[: self.rerank_k]
            passages = [f"{hit.title}. {hit.text}" for hit in candidates]
            scores = self.reranker.score(query, passages)
            if len(scores) != len(candidates):
                raise ValueError("Reranker score count does not match candidate count")
            for hit, score in zip(candidates, scores, strict=True):
                hit.reranker_score = round(score, 6)
                hit.score = hit.reranker_score
            fused = sorted(candidates, key=lambda hit: (-float(hit.reranker_score), -float(hit.rrf_score), hit.chunk_id))
        hits = apply_paper_diversity(fused, limit, self.max_per_paper)
        return SearchResponse(query=query, expanded_query=bm25_response.expanded_query, hits=hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search over the SC corpus")
    parser.add_argument("query")
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("artifacts/dense/embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/dense/metadata.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-provider", choices=("local", "api"), default="local")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--rerank-k", type=int, default=30)
    parser.add_argument("--max-per-paper", type=int, default=2)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    embedder = create_embedder(args.embedding_provider, args.embedding_model)
    dense = DenseIndex.from_artifacts(args.chunks, args.matrix, args.metadata, embedder)
    reranker = None if args.no_reranker else CrossEncoderReranker(args.reranker_model)
    retriever = HybridRetriever(
        BM25Index.from_jsonl(args.chunks),
        dense,
        reranker=reranker,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
        max_per_paper=args.max_per_paper,
    )
    response = retriever.search(args.query, args.limit)
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
