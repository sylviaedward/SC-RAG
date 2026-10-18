#!/usr/bin/env python3
"""Measure corpus retrieval against the paper aliases in the draft benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

from sc_rag.models import SearchResponse
from sc_rag.search import BM25Index


class Retriever(Protocol):
    name: str

    def search(self, query: str, limit: int = 8) -> SearchResponse: ...


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    index: Retriever,
    questions_path: Path,
    gold_path: Path,
    k: int,
    *,
    show_progress: bool = False,
) -> dict[str, object]:
    questions = {str(item["id"]): item for item in read_jsonl(questions_path)}
    gold = {str(item["id"]): item for item in read_jsonl(gold_path)}
    rows: list[dict[str, object]] = []
    reciprocal_ranks: list[float] = []
    any_hits = 0
    all_hits = 0
    latencies_ms: list[float] = []
    returned_hit_counts: list[int] = []

    for item_id, question in questions.items():
        aliases = set(gold[item_id]["gold_aliases"])
        if not question["answerable"] or not aliases:
            continue
        started = time.perf_counter()
        response = index.search(str(question["question"]), limit=k)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        retrieved = [hit.alias for hit in response.hits]
        returned_hit_counts.append(len(retrieved))
        ranks = [rank for rank, alias in enumerate(retrieved, start=1) if alias in aliases]
        any_match = bool(ranks)
        all_match = aliases <= set(retrieved)
        any_hits += int(any_match)
        all_hits += int(all_match)
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
        rows.append(
            {
                "id": item_id,
                "category": question["category"],
                "gold_aliases": sorted(aliases),
                "retrieved_aliases": retrieved,
                "any_hit": any_match,
                "all_hit": all_match,
                "returned_hits": len(retrieved),
            }
        )
        if show_progress and len(rows) % 5 == 0:
            print(f"{index.name}: evaluated {len(rows)}/50 questions", flush=True)

    count = len(rows)
    ordered_latencies = sorted(latencies_ms)
    p95_index = min(len(ordered_latencies) - 1, int(len(ordered_latencies) * 0.95))
    return {
        "retriever": index.name,
        "k": k,
        "evaluated_questions": count,
        "hit_rate_any": round(any_hits / count, 4),
        "hit_rate_all": round(all_hits / count, 4),
        "mrr": round(sum(reciprocal_ranks) / count, 4),
        "mean_returned_hits": round(sum(returned_hit_counts) / count, 2),
        "full_k_rate": round(sum(value == k for value in returned_hit_counts) / count, 4),
        "mean_latency_ms": round(sum(latencies_ms) / count, 2),
        "p95_latency_ms": round(ordered_latencies[p95_index], 2),
        "note": "Draft baseline; benchmark gold and page references still require domain review.",
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_answers.jsonl"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_eval.json"))
    args = parser.parse_args()
    report = evaluate(BM25Index.from_jsonl(args.index), args.questions, args.gold, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

