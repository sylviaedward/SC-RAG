#!/usr/bin/env python3
"""Run Phase 1 retrieval ablations with shared model instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_retrieval import evaluate
from sc_rag.dense import DEFAULT_EMBEDDING_MODEL, DenseIndex, create_embedder
from sc_rag.hybrid import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, HybridRetriever
from sc_rag.search import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("artifacts/dense/embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/dense/metadata.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_answers.jsonl"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-provider", choices=("local", "api"), default="local")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase1_eval.json"))
    args = parser.parse_args()

    embedder = create_embedder(args.embedding_provider, args.embedding_model)
    bm25 = BM25Index.from_jsonl(args.chunks)
    dense = DenseIndex.from_artifacts(args.chunks, args.matrix, args.metadata, embedder)
    retrievers = [bm25, dense, HybridRetriever(bm25, dense)]
    if not args.skip_reranker:
        retrievers.append(HybridRetriever(bm25, dense, reranker=CrossEncoderReranker(args.reranker_model)))

    reports = [evaluate(retriever, args.questions, args.gold, args.k) for retriever in retrievers]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_keys = (
        "retriever",
        "hit_rate_any",
        "hit_rate_all",
        "mrr",
        "mean_latency_ms",
        "p95_latency_ms",
    )
    print(json.dumps([{key: report[key] for key in summary_keys} for report in reports], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
