#!/usr/bin/env python3
"""Run the complete answerable/unanswerable regression and save an auditable report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from analyze_phase2_failures import reason_code


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * fraction))
    return round(values[index], 2)


def run_evaluation(args: argparse.Namespace, raw_output: Path) -> None:
    command = [
        sys.executable,
        "scripts/evaluate_phase2.py",
        "--scope",
        "all",
        "--output",
        str(raw_output),
        "--embedding-provider",
        args.embedding_provider,
        "--embedding-model",
        args.embedding_model,
    ]
    subprocess.run(command, check=True)


def build_report(raw: dict[str, object], *, started_at: float, embedding_provider: str, embedding_model: str) -> dict[str, object]:
    rows = list(raw["results"])
    enriched = []
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    latencies: list[float] = []
    for row in rows:
        response = row["response"]
        reason = reason_code(row)
        latency = float(row.get("latency_ms", 0))
        latencies.append(latency)
        category = str(row["category"])
        category_counts[category]["questions"] += 1
        category_counts[category]["usable"] += int(row["status"] in {"answered", "partial"})
        category_counts[category]["abstained"] += int(row["status"] == "abstained")
        enriched.append(
            {
                "id": row["id"],
                "category": category,
                "question": response.get("question", ""),
                "status": row["status"],
                "failure_reason": reason,
                "latency_ms": latency,
                "retrieval_trace": {
                    "candidate_chunks": response.get("candidate_chunks", 0),
                    "retrieved_chunks": response.get("retrieved_chunks", 0),
                    "evidence_aliases": response.get("evidence_aliases", []),
                    "requested_aliases": response.get("requested_aliases", []),
                    "missing_requested_aliases": response.get("missing_requested_aliases", []),
                },
                "generation_trace": {
                    "model": response.get("model"),
                    "evidence_sufficiency": response.get("evidence_sufficiency"),
                    "claims": len(response.get("claims", [])),
                    "generation_attempts": response.get("generation_attempts", 1),
                    "repair_reason": response.get("repair_reason"),
                },
            }
        )
    answerable = [row for row in rows if row["expected_answerable"]]
    unanswerable = [row for row in rows if not row["expected_answerable"]]
    usable = {"answered", "partial"}
    return {
        "phase": 4,
        "prompt_version": raw.get("prompt_version"),
        "model": raw.get("model"),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "evaluated_questions": len(rows),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "usable_response_rate_on_answerable": round(sum(row["status"] in usable for row in answerable) / len(answerable), 4) if answerable else None,
        "abstention_rate_on_unanswerable": round(sum(row["status"] == "abstained" for row in unanswerable) / len(unanswerable), 4) if unanswerable else None,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "failure_reason_counts": dict(Counter(item["failure_reason"] for item in enriched)),
        "category_summary": {key: dict(value) for key, value in sorted(category_counts.items())},
        "generated_at_epoch": round(time.time(), 3),
        "duration_seconds": round(time.time() - started_at, 2),
        "results": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase4_eval.json"))
    parser.add_argument("--embedding-provider", default="local")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="sc-rag-phase4-") as directory:
        raw_path = Path(directory) / "raw.json"
        run_evaluation(args, raw_path)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    report = build_report(raw, started_at=started, embedding_provider=args.embedding_provider, embedding_model=args.embedding_model)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
