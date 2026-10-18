#!/usr/bin/env python3
"""Evaluate grounded answers and abstention without exposing gold data to the model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sc_rag.answering import GroundedAnswerService, OpenAICompatibleChatClient, build_hybrid_retriever
from sc_rag.models import AnswerMode, AnswerStatus


PROMPT_VERSION = "phase2-grounded-v1"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mode_for_category(category: str) -> AnswerMode:
    if category == "fundamentals":
        return AnswerMode.fundamentals
    if category == "cross_paper":
        return AnswerMode.comparison
    if category in {"single_paper", "implementation"}:
        return AnswerMode.paper_lookup
    return AnswerMode.auto


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_answers.jsonl"))
    parser.add_argument("--scope", choices=("answerable", "unanswerable", "all"), default="unanswerable")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2_eval.json"))
    parser.add_argument("--embedding-provider", choices=("local", "api"), default="local")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    gold = {str(item["id"]): item for item in read_jsonl(args.gold)}
    selected = []
    requested_ids = set(args.ids)
    for question in questions:
        if requested_ids and question["id"] not in requested_ids:
            continue
        if args.scope == "answerable" and not question["answerable"]:
            continue
        if args.scope == "unanswerable" and question["answerable"]:
            continue
        selected.append(question)
    if not selected:
        raise ValueError("No questions matched the requested scope or IDs")

    service = GroundedAnswerService(
        build_hybrid_retriever(
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
        ),
        OpenAICompatibleChatClient.from_env(),
    )
    rows = []
    latencies = []
    for index, question in enumerate(selected, start=1):
        started = time.perf_counter()
        response = service.answer(str(question["question"]), mode_for_category(str(question["category"])))
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        expected_answerable = bool(question["answerable"])
        cited_aliases = sorted(
            {
                citation.alias
                for claim in response.claims
                for citation in claim.citations
            }
        )
        gold_aliases = set(gold[str(question["id"])]["gold_aliases"]) if expected_answerable else set()
        row = {
            "id": question["id"],
            "category": question["category"],
            "expected_answerable": expected_answerable,
            "status": response.status.value,
            "behavior_correct": (response.status == AnswerStatus.answered) == expected_answerable,
            "cited_aliases": cited_aliases,
            "gold_hit_any": bool(gold_aliases & set(cited_aliases)) if expected_answerable else None,
            "gold_hit_all": gold_aliases <= set(cited_aliases) if expected_answerable else None,
            "latency_ms": round(latency_ms, 2),
            "response": response.model_dump(mode="json"),
        }
        rows.append(row)
        report = build_report(args.scope, service.client.model_name, rows, latencies)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"evaluated {index}/{len(selected)}: {question['id']} -> {response.status.value}", flush=True)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


def build_report(scope: str, model: str, rows: list[dict[str, object]], latencies: list[float]) -> dict[str, object]:
    answerable = [row for row in rows if row["expected_answerable"]]
    unanswerable = [row for row in rows if not row["expected_answerable"]]
    answered = [row for row in rows if row["status"] == AnswerStatus.answered.value]
    return {
        "scope": scope,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "evaluated_questions": len(rows),
        "behavior_accuracy": round(sum(bool(row["behavior_correct"]) for row in rows) / len(rows), 4),
        "answer_rate_on_answerable": round(
            sum(row["status"] == AnswerStatus.answered.value for row in answerable) / len(answerable), 4
        ) if answerable else None,
        "abstention_rate_on_unanswerable": round(
            sum(row["status"] == AnswerStatus.abstained.value for row in unanswerable) / len(unanswerable), 4
        ) if unanswerable else None,
        "citation_gold_hit_any": round(
            sum(bool(row["gold_hit_any"]) for row in answerable) / len(answerable), 4
        ) if answerable else None,
        "citation_gold_hit_all": round(
            sum(bool(row["gold_hit_all"]) for row in answerable) / len(answerable), 4
        ) if answerable else None,
        "mean_claims_when_answered": round(
            sum(len(row["response"]["claims"]) for row in answered) / len(answered), 2
        ) if answered else 0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "results": rows,
    }


if __name__ == "__main__":
    main()
