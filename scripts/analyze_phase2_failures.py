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
    "claim_referenced_unknown_span",
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
    repairs = Counter(
        str(row["response"].get("repair_reason"))
        for row in rows
        if row["response"].get("repair_reason")
    )
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    failures: dict[str, list[str]] = defaultdict(list)
    accepted_rows = []
    for row in rows:
        category = str(row["category"])
        bucket = categories[category]
        bucket["questions"] += 1
        if row["status"] in {"answered", "partial"}:
            bucket["answered"] += 1
            targets = row.get("explicit_target_aliases", [])
            if targets:
                bucket["explicit_target_questions"] += 1
                bucket["explicit_target_all"] += int(bool(row.get("explicit_target_citation_hit_all")))
            bucket["reference_hit_any"] += int(
                bool(row.get("reference_source_hit_any", row.get("gold_hit_any")))
            )
            recall = row.get("reference_source_recall")
            if recall is None:
                recall = float(bool(row.get("gold_hit_all")))
            bucket["reference_recall_sum"] += float(recall)
            accepted_rows.append(row)
        else:
            failures[reason_code(row)].append(str(row["id"]))

    return {
        "evaluated_questions": len(rows),
        "accepted_answers": len(accepted_rows),
        "full_answers": sum(row["status"] == "answered" for row in accepted_rows),
        "partial_answers": sum(row["status"] == "partial" for row in accepted_rows),
        "conditional_reference_source_hit_any": round(
            sum(bool(row.get("reference_source_hit_any", row.get("gold_hit_any"))) for row in accepted_rows)
            / len(accepted_rows), 4
        ) if accepted_rows else None,
        "conditional_mean_reference_source_recall": round(
            sum(
                float(row["reference_source_recall"])
                if row.get("reference_source_recall") is not None
                else float(bool(row.get("gold_hit_all")))
                for row in accepted_rows
            ) / len(accepted_rows), 4
        ) if accepted_rows else None,
        "reason_counts": dict(reasons.most_common()),
        "repair_counts": dict(repairs.most_common()),
        "categories": dict(sorted(categories.items())),
        "failure_ids": dict(sorted(failures.items())),
    }


def render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Phase 2 failure analysis",
        "",
        f"Accepted {analysis['accepted_answers']} of {analysis['evaluated_questions']} answers.",
        f"Full answers: {analysis['full_answers']}; partial answers: {analysis['partial_answers']}.",
        f"Conditional reference-source Hit-any: {analysis['conditional_reference_source_hit_any']}",
        f"Conditional mean reference-source recall: {analysis['conditional_mean_reference_source_recall']}",
        "",
        "## Outcome reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in analysis["reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Generation repairs", "", "| Repair trigger | Count |", "|---|---:|"])
    for reason, count in analysis["repair_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Category breakdown",
            "",
            "| Category | Questions | Answered | Explicit targets all | Reference any | Mean reference recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, values in analysis["categories"].items():
        lines.append(
            f"| {category} | {values['questions']} | {values['answered']} | "
            f"{values['explicit_target_all']}/{values['explicit_target_questions']} | "
            f"{values['reference_hit_any']} | "
            f"{round(values['reference_recall_sum'] / values['answered'], 4) if values['answered'] else None} |"
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
