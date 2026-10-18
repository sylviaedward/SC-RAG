#!/usr/bin/env python3
"""Compare Hybrid RRF and reranked Phase 1 reports without rerunning models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_report(path: Path, retriever: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload if isinstance(payload, list) else [payload]
    for report in reports:
        if report.get("retriever") == retriever:
            return report
    names = ", ".join(str(report.get("retriever")) for report in reports)
    raise ValueError(f"{retriever!r} not found in {path}; available: {names}")


def reciprocal_rank(row: dict[str, object]) -> float:
    gold = set(row["gold_aliases"])
    for rank, alias in enumerate(row["retrieved_aliases"], start=1):
        if alias in gold:
            return 1.0 / rank
    return 0.0


def gold_coverage(row: dict[str, object]) -> int:
    return len(set(row["gold_aliases"]) & set(row["retrieved_aliases"]))


def compare_reports(base: dict[str, object], reranked: dict[str, object]) -> dict[str, object]:
    base_rows = {str(row["id"]): row for row in base["results"]}
    reranked_rows = {str(row["id"]): row for row in reranked["results"]}
    if base_rows.keys() != reranked_rows.keys():
        missing = sorted(base_rows.keys() - reranked_rows.keys())
        extra = sorted(reranked_rows.keys() - base_rows.keys())
        raise ValueError(f"Report question sets differ; missing={missing}, extra={extra}")

    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"questions": 0, "rr_improved": 0, "rr_harmed": 0, "coverage_improved": 0, "coverage_harmed": 0}
    )
    rows = []
    for item_id, base_row in base_rows.items():
        reranked_row = reranked_rows[item_id]
        before_rr = reciprocal_rank(base_row)
        after_rr = reciprocal_rank(reranked_row)
        before_coverage = gold_coverage(base_row)
        after_coverage = gold_coverage(reranked_row)
        category = str(base_row["category"])
        bucket = categories[category]
        bucket["questions"] += 1
        bucket["rr_improved"] += int(after_rr > before_rr)
        bucket["rr_harmed"] += int(after_rr < before_rr)
        bucket["coverage_improved"] += int(after_coverage > before_coverage)
        bucket["coverage_harmed"] += int(after_coverage < before_coverage)
        rows.append(
            {
                "id": item_id,
                "category": category,
                "gold_aliases": base_row["gold_aliases"],
                "base_reciprocal_rank": round(before_rr, 4),
                "reranked_reciprocal_rank": round(after_rr, 4),
                "reciprocal_rank_delta": round(after_rr - before_rr, 4),
                "base_gold_coverage": before_coverage,
                "reranked_gold_coverage": after_coverage,
                "gold_coverage_delta": after_coverage - before_coverage,
                "base_retrieved_aliases": base_row["retrieved_aliases"],
                "reranked_retrieved_aliases": reranked_row["retrieved_aliases"],
            }
        )

    return {
        "base_retriever": base["retriever"],
        "reranked_retriever": reranked["retriever"],
        "question_count": len(rows),
        "rr_improved": sum(row["reciprocal_rank_delta"] > 0 for row in rows),
        "rr_harmed": sum(row["reciprocal_rank_delta"] < 0 for row in rows),
        "coverage_improved": sum(row["gold_coverage_delta"] > 0 for row in rows),
        "coverage_harmed": sum(row["gold_coverage_delta"] < 0 for row in rows),
        "any_hit_gained": sum(not base_rows[row["id"]]["any_hit"] and reranked_rows[row["id"]]["any_hit"] for row in rows),
        "any_hit_lost": sum(base_rows[row["id"]]["any_hit"] and not reranked_rows[row["id"]]["any_hit"] for row in rows),
        "all_hit_gained": sum(not base_rows[row["id"]]["all_hit"] and reranked_rows[row["id"]]["all_hit"] for row in rows),
        "all_hit_lost": sum(base_rows[row["id"]]["all_hit"] and not reranked_rows[row["id"]]["all_hit"] for row in rows),
        "categories": dict(sorted(categories.items())),
        "results": rows,
    }


def render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Phase 1 reranker failure analysis",
        "",
        f"Compared `{analysis['base_retriever']}` with `{analysis['reranked_retriever']}` on {analysis['question_count']} questions.",
        "",
        "| Signal | Improved/gained | Harmed/lost |",
        "|---|---:|---:|",
        f"| Reciprocal rank | {analysis['rr_improved']} | {analysis['rr_harmed']} |",
        f"| Gold-paper coverage | {analysis['coverage_improved']} | {analysis['coverage_harmed']} |",
        f"| Any-hit status | {analysis['any_hit_gained']} | {analysis['any_hit_lost']} |",
        f"| All-hit status | {analysis['all_hit_gained']} | {analysis['all_hit_lost']} |",
        "",
        "## Category breakdown",
        "",
        "| Category | Questions | RR + | RR - | Coverage + | Coverage - |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in analysis["categories"].items():
        lines.append(
            f"| {category} | {values['questions']} | {values['rr_improved']} | {values['rr_harmed']} | "
            f"{values['coverage_improved']} | {values['coverage_harmed']} |"
        )

    changed = sorted(
        (row for row in analysis["results"] if row["reciprocal_rank_delta"] or row["gold_coverage_delta"]),
        key=lambda row: (row["gold_coverage_delta"], row["reciprocal_rank_delta"]),
    )
    lines.extend(
        [
            "",
            "## Changed questions",
            "",
            "Negative deltas appear first. Inspect these IDs in the benchmark and the two JSON reports.",
            "",
            "| ID | Category | RR delta | Coverage delta | Gold aliases |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in changed:
        aliases = ", ".join(row["gold_aliases"])
        lines.append(
            f"| {row['id']} | {row['category']} | {row['reciprocal_rank_delta']:+.4f} | "
            f"{row['gold_coverage_delta']:+d} | {aliases} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=Path("artifacts/phase1_core_eval.json"))
    parser.add_argument("--reranker", type=Path, default=Path("artifacts/phase1_reranker_eval.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase1_failure_analysis.md"))
    args = parser.parse_args()

    base = load_report(args.core, "hybrid_rrf")
    reranked = load_report(args.reranker, "hybrid_rrf_reranked")
    analysis = compare_reports(base, reranked)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({key: value for key, value in analysis.items() if key not in {"categories", "results"}}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
