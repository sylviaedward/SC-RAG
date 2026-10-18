#!/usr/bin/env python3
"""Extract the SC-RAG draft benchmark into isolated question and gold files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


CATEGORY_NAMES = {
    "A": "fundamentals",
    "B": "single_paper",
    "C": "cross_paper",
    "D": "implementation",
    "E": "unanswerable",
}
EXPECTED_COUNTS = {"A": 15, "B": 15, "C": 10, "D": 10, "E": 8}
ID_PATTERN = re.compile(r"(?m)^[ \t]*([A-E]\d+)\.\s*")
SECTION_PATTERN = re.compile(r"^[A-E]\.\s+.*(?:问题|参考答案).*$")
ALIAS_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-raw", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return result.stdout.replace("\f", "\n")


def clean_block(value: str) -> str:
    kept: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            kept.append("")
            continue
        if line == "随机计算（SC）RAG 知识库评测题库":
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if SECTION_PATTERN.match(line):
            continue
        if line.startswith("第一部分") or line.startswith("第二部分"):
            continue
        kept.append(line)

    joined = " ".join(part for part in kept if part)
    for boundary_note in (
        "说明：以下问题在这 50 篇语料中没有可支撑答案的信息",
        "以下结论均通过对全部 50 个文本文件做关键词检索核查",
        "附：审核与冻结建议",
    ):
        if boundary_note in joined:
            joined = joined.split(boundary_note, 1)[0]
    return re.sub(r"\s+", " ", joined).strip()


def segment(blob: str) -> dict[str, str]:
    matches = list(ID_PATTERN.finditer(blob))
    records: dict[str, str] = {}
    for index, match in enumerate(matches):
        item_id = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(blob)
        records[item_id] = clean_block(blob[match.end() : end])
    return records


def validate(records: dict[str, str], label: str) -> None:
    counts = {category: 0 for category in EXPECTED_COUNTS}
    for item_id, text in records.items():
        counts[item_id[0]] += 1
        if not text:
            raise SystemExit(f"Empty {label} record: {item_id}")
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected {label} counts: {counts}; expected {EXPECTED_COUNTS}")


def split_answer(value: str) -> tuple[str, str | None]:
    value = re.sub(r"审核：\[\s*\]\s*$", "", value).strip()
    # PDF line wrapping can split this label into "出\n处".
    value = re.sub(r"出\s*处：", "出处：", value)
    if "出处：" not in value:
        return value, None
    answer, citations = value.split("出处：", 1)
    return answer.strip(), citations.strip()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_answers.jsonl"))
    args = parser.parse_args()

    text = pdf_text(args.source)
    # Use the final occurrence because the PDF front matter contains a TOC copy.
    question_start = text.rfind("第一部分 题目卷")
    if question_start < 0:
        question_start = text.rfind("第一部分           题目卷")
    answer_start = text.find("第二部分", question_start + 1)
    if question_start < 0 or answer_start < 0:
        raise SystemExit("Could not locate question/answer boundaries")

    questions = segment(text[question_start:answer_start])
    answers = segment(text[answer_start:])
    validate(questions, "question")
    validate(answers, "answer")
    if questions.keys() != answers.keys():
        raise SystemExit("Question and answer IDs differ")

    question_records: list[dict[str, object]] = []
    gold_records: list[dict[str, object]] = []
    for item_id, question in questions.items():
        category = CATEGORY_NAMES[item_id[0]]
        answer, citation_text = split_answer(answers[item_id])
        aliases = []
        if citation_text:
            aliases = [alias for alias in ALIAS_PATTERN.findall(citation_text) if alias.strip()]
        question_records.append(
            {
                "id": item_id,
                "category": category,
                "question": question,
                "answerable": category != "unanswerable",
                "benchmark_status": "draft_needs_domain_review",
            }
        )
        gold_records.append(
            {
                "id": item_id,
                "answer": answer,
                "citation_text": citation_text,
                "gold_aliases": aliases,
                "review_status": "pending",
            }
        )

    write_jsonl(args.questions, question_records)
    write_jsonl(args.gold, gold_records)
    print(f"wrote {len(question_records)} questions and {len(gold_records)} gold answers")


if __name__ == "__main__":
    main()
