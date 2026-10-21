#!/usr/bin/env python3
"""Compare the existing hybrid retriever with and without its reranker.

This script intentionally reports measured values only. It uses the checked-in
58-question benchmark and requires the dense artifacts plus an LLM variable
group for the end-to-end answer metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sc_rag.answering import (
    GroundedAnswerService,
    OpenAICompatibleChatClient,
    load_known_aliases,
)
from sc_rag.dense import DEFAULT_EMBEDDING_MODEL, DenseIndex, create_embedder
from sc_rag.hybrid import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, HybridRetriever
from sc_rag.models import AnswerMode, AnswerStatus
from sc_rag.search import BM25Index


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))], 2)


def mode_for_category(category: str) -> AnswerMode:
    return {
        "fundamentals": AnswerMode.fundamentals,
        "cross_paper": AnswerMode.comparison,
        "single_paper": AnswerMode.paper_lookup,
        "implementation": AnswerMode.paper_lookup,
    }.get(category, AnswerMode.auto)


def run_group(name: str, retriever: HybridRetriever, questions: list[dict[str, object]], gold: dict[str, dict[str, object]], known_aliases: list[str], client: object) -> dict[str, object]:
    service = GroundedAnswerService(retriever, client, known_aliases=known_aliases)
    retrieval_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    top5_hits = 0
    answerable_correct = 0
    unanswerable_correct = 0
    answerable_count = 0
    unanswerable_count = 0

    for question in questions:
        question_id = str(question["id"])
        started = time.perf_counter()
        search = retriever.search(str(question["question"]), limit=5)
        retrieval_latencies.append((time.perf_counter() - started) * 1000)
        aliases = set(gold[question_id]["gold_aliases"])
        top5_hits += int(bool(aliases & {hit.alias for hit in search.hits}))

        started = time.perf_counter()
        response = service.answer(str(question["question"]), mode_for_category(str(question["category"])))
        end_to_end_latencies.append((time.perf_counter() - started) * 1000)
        usable = response.status in {AnswerStatus.answered, AnswerStatus.partial}
        if bool(question["answerable"]):
            answerable_count += 1
            answerable_correct += int(usable)
        else:
            unanswerable_count += 1
            unanswerable_correct += int(response.status == AnswerStatus.abstained)

    return {
        "name": name,
        "retriever": retriever.name,
        "evaluated_questions": len(questions),
        "top5_recall_any": round(top5_hits / answerable_count, 4) if answerable_count else None,
        "answerable_accuracy": round(answerable_correct / answerable_count, 4) if answerable_count else None,
        "abstain_correctness": round(unanswerable_correct / unanswerable_count, 4) if unanswerable_count else None,
        "retrieval_p50_latency_ms": percentile(retrieval_latencies, 0.50),
        "retrieval_p95_latency_ms": percentile(retrieval_latencies, 0.95),
        "end_to_end_p50_latency_ms": percentile(end_to_end_latencies, 0.50),
        "end_to_end_p95_latency_ms": percentile(end_to_end_latencies, 0.95),
        "measurement_note": "Measured locally on the checked-in 50 answerable + 8 unanswerable questions; do not hand-edit values.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("artifacts/dense/embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/dense/metadata.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_answers.jsonl"))
    parser.add_argument("--embedding-provider", choices=("local", "api"), default="local")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--output", type=Path, default=Path("reports/reranker_ablation.json"))
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    gold = {str(item["id"]): item for item in read_jsonl(args.gold)}
    answerable = [item for item in questions if item["answerable"]]
    if len(questions) != 58 or len(answerable) != 50:
        raise ValueError("Expected the checked-in 58-question benchmark (50 answerable + 8 unanswerable)")

    embedder = create_embedder(args.embedding_provider, args.embedding_model)
    dense = DenseIndex.from_artifacts(args.chunks, args.matrix, args.metadata, embedder)
    bm25 = BM25Index.from_jsonl(args.chunks)
    base = dict(
        bm25=bm25,
        dense=dense,
        candidate_k=40,
        max_per_paper=2,
    )
    retrievers = [
        ("no_reranker", HybridRetriever(**base)),
        ("reranker", HybridRetriever(**base, reranker=CrossEncoderReranker(args.reranker_model))),
    ]
    client = OpenAICompatibleChatClient.from_env()
    known_aliases = load_known_aliases(args.chunks)
    reports = [run_group(name, retriever, questions, gold, known_aliases, client) for name, retriever in retrievers]
    report = {
        "benchmark": {"questions": 58, "answerable": 50, "unanswerable": 8},
        "groups": reports,
        "measurement_note": "All values are generated by this script from local execution; no placeholder or hand-entered performance number is accepted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
