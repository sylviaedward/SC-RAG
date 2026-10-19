#!/usr/bin/env python3
"""Attribute citation coverage loss to retrieval or grounded generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_retriever_report(path: Path, retriever: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload if isinstance(payload, list) else [payload]
    for report in reports:
        if report.get("retriever") == retriever:
            return report
    available = ", ".join(str(report.get("retriever")) for report in reports)
    raise ValueError(f"{retriever!r} not found in {path}; available: {available}")


def attribute(retrieval_report: dict[str, object], answer_report: dict[str, object]) -> dict[str, object]:
    retrieval_rows = {str(row["id"]): row for row in retrieval_report["results"]}
    answer_rows = {str(row["id"]): row for row in answer_report["results"]}
    if retrieval_rows.keys() != answer_rows.keys():
        missing = sorted(retrieval_rows.keys() - answer_rows.keys())
        extra = sorted(answer_rows.keys() - retrieval_rows.keys())
        raise ValueError(f"Report question sets differ; missing_from_phase2={missing}, extra_in_phase2={extra}")

    categories: dict[str, Counter[str]] = defaultdict(Counter)
    causes: Counter[str] = Counter()
    rows = []
    for item_id, retrieval_row in retrieval_rows.items():
        answer_row = answer_rows[item_id]
        gold = set(retrieval_row["gold_aliases"])
        retrieved = set(retrieval_row["retrieved_aliases"])
        cited = set(answer_row["cited_aliases"])
        retrieval_any = bool(gold & retrieved)
        retrieval_all = gold <= retrieved
        citation_any = bool(gold & cited)
        citation_all = gold <= cited
        status = str(answer_row["status"])

        if not retrieval_any:
            cause = "retrieval_miss_all"
        elif not retrieval_all:
            cause = "retrieval_incomplete"
        elif status == "abstained":
            cause = "generation_abstained_with_full_retrieval"
        elif not citation_all:
            cause = "generation_omitted_retrieved_gold"
        else:
            cause = "full_coverage"

        category = str(answer_row["category"])
        bucket = categories[category]
        bucket["questions"] += 1
        bucket["retrieval_any"] += int(retrieval_any)
        bucket["retrieval_all"] += int(retrieval_all)
        bucket["citation_any"] += int(citation_any)
        bucket["citation_all"] += int(citation_all)
        bucket[cause] += 1
        causes[cause] += 1
        rows.append(
            {
                "id": item_id,
                "category": category,
                "status": status,
                "cause": cause,
                "gold_aliases": sorted(gold),
                "missing_from_retrieval": sorted(gold - retrieved),
                "retrieved_gold_not_cited": sorted((gold & retrieved) - cited),
                "cited_gold_aliases": sorted(gold & cited),
            }
        )

    count = len(rows)
    return {
        "question_count": count,
        "retriever": retrieval_report["retriever"],
        "prompt_version": answer_report.get("prompt_version"),
        "retrieval_hit_any": round(sum(bool(set(row["gold_aliases"]) - set(row["missing_from_retrieval"])) for row in rows) / count, 4),
        "retrieval_hit_all": round(sum(not row["missing_from_retrieval"] for row in rows) / count, 4),
        "citation_hit_any": round(sum(bool(row["cited_gold_aliases"]) for row in rows) / count, 4),
        "citation_hit_all": round(
            sum(not row["missing_from_retrieval"] and not row["retrieved_gold_not_cited"] for row in rows) / count,
            4,
        ),
        "cause_counts": dict(causes.most_common()),
        "categories": {category: dict(values) for category, values in sorted(categories.items())},
        "results": rows,
    }


def render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Phase 2 cross-layer attribution",
        "",
        f"Retriever: `{analysis['retriever']}`; prompt: `{analysis['prompt_version']}`.",
        "",
        "| Layer metric | Score |",
        "|---|---:|",
        f"| Retrieval Hit-any | {analysis['retrieval_hit_any']} |",
        f"| Retrieval Hit-all | {analysis['retrieval_hit_all']} |",
        f"| Final citation Hit-any | {analysis['citation_hit_any']} |",
        f"| Final citation Hit-all | {analysis['citation_hit_all']} |",
        "",
        "## Root causes",
        "",
        "| Cause | Questions |",
        "|---|---:|",
    ]
    for cause, count in analysis["cause_counts"].items():
        lines.append(f"| {cause} | {count} |")
    lines.extend(
        [
            "",
            "## Category breakdown",
            "",
            "| Category | N | Retrieval any | Retrieval all | Citation any | Citation all |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, values in analysis["categories"].items():
        lines.append(
            f"| {category} | {values['questions']} | {values['retrieval_any']} | {values['retrieval_all']} | "
            f"{values['citation_any']} | {values['citation_all']} |"
        )
    lines.extend(["", "## Questions requiring action", ""])
    for row in analysis["results"]:
        if row["cause"] == "full_coverage":
            continue
        lines.append(
            f"- `{row['id']}` `{row['cause']}`: missing from retrieval={row['missing_from_retrieval']}; "
            f"retrieved but not cited={row['retrieved_gold_not_cited']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, default=Path("artifacts/phase1_core_eval.json"))
    parser.add_argument("--phase2", type=Path, default=Path("artifacts/phase2_answerable_eval.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2_attribution.md"))
    args = parser.parse_args()
    retrieval = load_retriever_report(args.retrieval, "hybrid_rrf")
    answers = json.loads(args.phase2.read_text(encoding="utf-8"))
    analysis = attribute(retrieval, answers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({key: value for key, value in analysis.items() if key != "results"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
