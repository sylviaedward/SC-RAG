#!/usr/bin/env python3
"""Analyze Phase 2 abstentions and accepted-answer precision without API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


VALIDATION_REASONS = {
    "insufficient_output_contained_claims",
    "supported_output_contained_no_claims",
    "claim_contained_duplicate_citations",
    "claim_referenced_unknown_evidence",
    "citation_quote_not_found_in_evidence",
    "partial_output_contained_no_missing_information",
    "sufficient_output_contained_missing_information",
}


def reason_code(row: dict[str, object]) -> str:
    if row["status"] in {"answered", "partial"}:
        return f"accepted_{row['status']}"
    response = row["response"]
    reason = str(response["reason"])
    if reason.startswith("model_output_validation_failed"):
        return "model_output_validation_failed"
    if reason in VALIDATION_REASONS:
        return reason
    if response["evidence_sufficiency"] == "insufficient":
        return "model_declared_insufficient"
    if response["evidence_sufficiency"] == "conflicting":
        return "conflicting_output_rejected"
    return "other_abstention"


def analyze(report: dict[str, object]) -> dict[str, object]:
    rows = report["results"]
    reasons = Counter(reason_code(row) for row in rows)
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"questions": 0, "answered": 0, "gold_hit_any": 0, "gold_hit_all": 0}
    )
    failures: dict[str, list[str]] = defaultdict(list)
    accepted_rows = []
    for row in rows:
        category = str(row["category"])
        bucket = categories[category]
        bucket["questions"] += 1
        if row["status"] in {"answered", "partial"}:
            bucket["answered"] += 1
            bucket["gold_hit_any"] += int(bool(row["gold_hit_any"]))
            bucket["gold_hit_all"] += int(bool(row["gold_hit_all"]))
            accepted_rows.append(row)
        else:
            failures[reason_code(row)].append(str(row["id"]))

    return {
        "evaluated_questions": len(rows),
        "accepted_answers": len(accepted_rows),
        "full_answers": sum(row["status"] == "answered" for row in accepted_rows),
        "partial_answers": sum(row["status"] == "partial" for row in accepted_rows),
        "conditional_citation_gold_hit_any": round(
            sum(bool(row["gold_hit_any"]) for row in accepted_rows) / len(accepted_rows), 4
        ) if accepted_rows else None,
        "conditional_citation_gold_hit_all": round(
            sum(bool(row["gold_hit_all"]) for row in accepted_rows) / len(accepted_rows), 4
        ) if accepted_rows else None,
        "reason_counts": dict(reasons.most_common()),
        "categories": dict(sorted(categories.items())),
        "failure_ids": dict(sorted(failures.items())),
    }


def render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Phase 2 failure analysis",
        "",
        f"Accepted {analysis['accepted_answers']} of {analysis['evaluated_questions']} answers.",
        f"Full answers: {analysis['full_answers']}; partial answers: {analysis['partial_answers']}.",
        f"Conditional citation gold Hit-any: {analysis['conditional_citation_gold_hit_any']}",
        f"Conditional citation gold Hit-all: {analysis['conditional_citation_gold_hit_all']}",
        "",
        "## Outcome reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in analysis["reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Category breakdown",
            "",
            "| Category | Questions | Answered | Gold any | Gold all |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, values in analysis["categories"].items():
        lines.append(
            f"| {category} | {values['questions']} | {values['answered']} | "
            f"{values['gold_hit_any']} | {values['gold_hit_all']} |"
        )
    lines.extend(["", "## Failed question IDs", ""])
    for reason, ids in analysis["failure_ids"].items():
        lines.append(f"- `{reason}`: {', '.join(ids)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/phase2_answerable_eval.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2_failure_analysis.md"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({key: value for key, value in analysis.items() if key != "failure_ids"}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
