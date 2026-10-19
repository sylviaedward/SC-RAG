#!/usr/bin/env python3
"""Attribute explicit-target and reference-source loss across planning and generation."""

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
    """Use v4's actual evidence plan, with Phase 1 Top-K as a legacy fallback."""
    retrieval_rows = {str(row["id"]): row for row in retrieval_report["results"]}
    answer_rows = {str(row["id"]): row for row in answer_report["results"]}
    missing_retrieval_rows = sorted(answer_rows.keys() - retrieval_rows.keys())
    if missing_retrieval_rows:
        raise ValueError(f"Questions missing from retrieval report: {missing_retrieval_rows}")

    categories: dict[str, Counter[str]] = defaultdict(Counter)
    causes: Counter[str] = Counter()
    rows = []
    used_planner_trace = False
    for item_id, answer_row in answer_rows.items():
        retrieval_row = retrieval_rows[item_id]
        response = answer_row.get("response", {})
        if "evidence_aliases" in response:
            evidence = set(response["evidence_aliases"])
            used_planner_trace = True
        else:
            evidence = set(retrieval_row["retrieved_aliases"])

        reference = set(answer_row.get("reference_aliases", retrieval_row["gold_aliases"]))
        targets = set(answer_row.get("explicit_target_aliases", response.get("requested_aliases", [])))
        cited = set(answer_row["cited_aliases"])
        reference_evidence_hits = reference & evidence
        reference_citation_hits = reference & cited
        target_evidence_all = targets <= evidence if targets else None
        target_citation_all = targets <= cited if targets else None
        status = str(answer_row["status"])

        if targets and not target_evidence_all:
            cause = "explicit_target_missing_from_evidence"
        elif targets and not target_citation_all:
            cause = "generation_omitted_explicit_target"
        elif targets:
            cause = "explicit_target_covered"
        elif not reference_evidence_hits:
            cause = "reference_retrieval_miss_all"
        elif status == "abstained":
            cause = "generation_abstained_with_reference_evidence"
        elif not reference_citation_hits:
            cause = "reference_source_not_cited"
        else:
            cause = "reference_source_covered"

        category = str(answer_row["category"])
        bucket = categories[category]
        bucket["questions"] += 1
        bucket["explicit_target_questions"] += int(bool(targets))
        bucket["explicit_target_evidence_all"] += int(bool(target_evidence_all))
        bucket["explicit_target_citation_all"] += int(bool(target_citation_all))
        bucket["reference_evidence_any"] += int(bool(reference_evidence_hits))
        bucket["reference_citation_any"] += int(bool(reference_citation_hits))
        bucket[cause] += 1
        causes[cause] += 1
        rows.append(
            {
                "id": item_id,
                "category": category,
                "status": status,
                "cause": cause,
                "explicit_target_aliases": sorted(targets),
                "missing_targets_from_evidence": sorted(targets - evidence),
                "evidence_targets_not_cited": sorted((targets & evidence) - cited),
                "reference_aliases": sorted(reference),
                "missing_all_reference_evidence": not bool(reference_evidence_hits),
                "cited_reference_aliases": sorted(reference_citation_hits),
                "cited_non_reference_aliases": sorted(cited - reference),
            }
        )

    count = len(rows)
    target_rows = [row for row in rows if row["explicit_target_aliases"]]
    return {
        "question_count": count,
        "evidence_source": "phase2_alias_planner" if used_planner_trace else retrieval_report["retriever"],
        "prompt_version": answer_report.get("prompt_version"),
        "explicit_target_question_count": len(target_rows),
        "explicit_target_evidence_hit_all": round(
            sum(not row["missing_targets_from_evidence"] for row in target_rows) / len(target_rows), 4
        ) if target_rows else None,
        "explicit_target_citation_hit_all": round(
            sum(
                not row["missing_targets_from_evidence"] and not row["evidence_targets_not_cited"]
                for row in target_rows
            ) / len(target_rows), 4
        ) if target_rows else None,
        "reference_evidence_hit_any": round(
            sum(not row["missing_all_reference_evidence"] for row in rows) / count, 4
        ),
        "reference_citation_hit_any": round(
            sum(bool(row["cited_reference_aliases"]) for row in rows) / count, 4
        ),
        "cause_counts": dict(causes.most_common()),
        "categories": {category: dict(values) for category, values in sorted(categories.items())},
        "results": rows,
    }


def render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Phase 2 cross-layer attribution",
        "",
        f"Evidence source: `{analysis['evidence_source']}`; prompt: `{analysis['prompt_version']}`.",
        "",
        "| Layer metric | Score |",
        "|---|---:|",
        f"| Explicit-target evidence Hit-all | {analysis['explicit_target_evidence_hit_all']} |",
        f"| Explicit-target citation Hit-all | {analysis['explicit_target_citation_hit_all']} |",
        f"| Reference-source evidence Hit-any | {analysis['reference_evidence_hit_any']} |",
        f"| Reference-source citation Hit-any | {analysis['reference_citation_hit_any']} |",
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
            "| Category | N | Target N | Target evidence all | Target citation all | Reference evidence any | Reference citation any |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category, values in analysis["categories"].items():
        lines.append(
            f"| {category} | {values['questions']} | {values['explicit_target_questions']} | "
            f"{values['explicit_target_evidence_all']} | {values['explicit_target_citation_all']} | "
            f"{values['reference_evidence_any']} | {values['reference_citation_any']} |"
        )
    lines.extend(["", "## Questions requiring action", ""])
    for row in analysis["results"]:
        if row["cause"] in {"explicit_target_covered", "reference_source_covered"}:
            continue
        lines.append(
            f"- `{row['id']}` `{row['cause']}`: missing targets from evidence="
            f"{row['missing_targets_from_evidence']}; evidence targets not cited="
            f"{row['evidence_targets_not_cited']}; cited non-reference sources="
            f"{row['cited_non_reference_aliases']}"
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
