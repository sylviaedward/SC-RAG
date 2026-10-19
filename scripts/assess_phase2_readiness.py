#!/usr/bin/env python3
"""Apply Phase 2 release gates to complete answerable and unanswerable reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


USABLE_STATUSES = {"answered", "partial"}


def read_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def citation_contract_valid(row: dict[str, object]) -> bool:
    response = row.get("response", {})
    claims = response.get("claims", [])
    if not claims:
        return False
    return all(
        claim.get("citations")
        and all(
            citation.get("span_id")
            and citation.get("chunk_id")
            and citation.get("alias")
            and citation.get("excerpt")
            for citation in claim["citations"]
        )
        for claim in claims
    )


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def assess(
    answerable_report: dict[str, object],
    unanswerable_report: dict[str, object],
    *,
    expected_answerable: int = 50,
    expected_unanswerable: int = 8,
    min_usable: float = 0.90,
    min_target_evidence: float = 0.90,
    min_target_citation: float = 0.85,
    min_abstention: float = 0.875,
) -> dict[str, object]:
    answerable = [row for row in answerable_report["results"] if row.get("expected_answerable") is True]
    unanswerable = [row for row in unanswerable_report["results"] if row.get("expected_answerable") is False]
    target_rows = [row for row in answerable if row.get("explicit_target_aliases")]
    usable_rows = [row for row in answerable if row.get("status") in USABLE_STATUSES]
    contract_violations = [row for row in usable_rows if not citation_contract_valid(row)]

    usable_rate = ratio(len(usable_rows), len(answerable))
    target_evidence_rate = ratio(
        sum(bool(row.get("explicit_target_evidence_hit_all")) for row in target_rows),
        len(target_rows),
    )
    target_citation_rate = ratio(
        sum(bool(row.get("explicit_target_citation_hit_all")) for row in target_rows),
        len(target_rows),
    )
    abstention_rate = ratio(
        sum(row.get("status") == "abstained" for row in unanswerable),
        len(unanswerable),
    )
    alternative_sources = [
        row for row in usable_rows
        if row.get("reference_source_hit_any") is False and citation_contract_valid(row)
    ]
    answerable_ids = [str(row["id"]) for row in answerable]
    unanswerable_ids = [str(row["id"]) for row in unanswerable]
    same_experiment = (
        bool(answerable_report.get("prompt_version"))
        and answerable_report.get("prompt_version") == unanswerable_report.get("prompt_version")
        and bool(answerable_report.get("model"))
        and answerable_report.get("model") == unanswerable_report.get("model")
    )

    checks = {
        "complete_answerable_set": (
            len(answerable) == expected_answerable and len(set(answerable_ids)) == expected_answerable
        ),
        "complete_unanswerable_set": (
            len(unanswerable) == expected_unanswerable and len(set(unanswerable_ids)) == expected_unanswerable
        ),
        "matching_experiment_version": same_experiment,
        "usable_response_rate": usable_rate is not None and usable_rate >= min_usable,
        "explicit_target_evidence_hit_all": (
            target_evidence_rate is not None and target_evidence_rate >= min_target_evidence
        ),
        "explicit_target_citation_hit_all": (
            target_citation_rate is not None and target_citation_rate >= min_target_citation
        ),
        "unanswerable_abstention_rate": (
            abstention_rate is not None and abstention_rate >= min_abstention
        ),
        "accepted_citation_contract_violations": not contract_violations,
    }
    return {
        "phase": "phase2",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "expected_answerable": expected_answerable,
            "expected_unanswerable": expected_unanswerable,
            "min_usable_response_rate": min_usable,
            "min_explicit_target_evidence_hit_all": min_target_evidence,
            "min_explicit_target_citation_hit_all": min_target_citation,
            "min_unanswerable_abstention_rate": min_abstention,
        },
        "metrics": {
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),
            "explicit_target_questions": len(target_rows),
            "usable_response_rate": usable_rate,
            "explicit_target_evidence_hit_all": target_evidence_rate,
            "explicit_target_citation_hit_all": target_citation_rate,
            "unanswerable_abstention_rate": abstention_rate,
            "accepted_citation_contract_violations": len(contract_violations),
        },
        "diagnostics_not_gates": {
            "reference_source_hit_any": answerable_report.get("reference_source_hit_any"),
            "mean_reference_source_recall": answerable_report.get("mean_reference_source_recall"),
            "grounded_alternative_source_count": len(alternative_sources),
            "grounded_alternative_source_ids": [str(row["id"]) for row in alternative_sources],
        },
        "versions": {
            "answerable_prompt": answerable_report.get("prompt_version"),
            "unanswerable_prompt": unanswerable_report.get("prompt_version"),
            "answerable_model": answerable_report.get("model"),
            "unanswerable_model": unanswerable_report.get("model"),
        },
    }


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Phase 2 readiness",
        "",
        f"**Passed:** `{str(result['passed']).lower()}`",
        "",
        "## Release gates",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"| `{name}` | {str(passed).lower()} |")
    lines.extend(["", "## Metrics", "", "```json", json.dumps(result["metrics"], ensure_ascii=False, indent=2), "```"])
    lines.extend(
        [
            "",
            "## Diagnostics that do not block Phase 3",
            "",
            "```json",
            json.dumps(result["diagnostics_not_gates"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerable", type=Path, default=Path("artifacts/phase2_answerable_eval.json"))
    parser.add_argument("--unanswerable", type=Path, default=Path("artifacts/phase2_unanswerable_eval.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2_readiness.json"))
    parser.add_argument("--markdown", type=Path, default=Path("artifacts/phase2_readiness.md"))
    args = parser.parse_args()

    result = assess(read_report(args.answerable), read_report(args.unanswerable))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
