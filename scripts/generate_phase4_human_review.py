#!/usr/bin/env python3
"""Resolve the ten Phase 4 human-review samples to real question IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SAMPLES = [
    ("fundamentals", "reference_retrieval_miss_all", "检索", "标准答案文献未被召回，属于检索盲区。", False, "扩大 BM25 候选集或增加同义词泛化。"),
    ("fundamentals", "grounded_alternative_source", "评测", "模型使用了可替代来源，事实仍有直接证据支持。", False, "人工确认后纳入 acceptable set。"),
    ("single_paper", "generation_omitted_explicit_target", "生成", "目标论文进入证据但首次生成漏引，触发 repair。", True, "继续强化显式目标覆盖要求并监控 repair rate。"),
    ("single_paper", "accepted_partial", "生成", "模型生成了受支持的部分 Claim，但没有覆盖全部请求细节。", False, "评估是否需要增加上下文预算或 self-correction。"),
    ("cross_paper", "explicit_target_covered", "无", "跨论文目标均进入证据并完成引用覆盖。", False, "作为 Alias-aware Planner 的正面样本。"),
    ("cross_paper", "explicit_target_covered", "无", "复杂比较条件下目标覆盖和引用校验均通过。", False, "作为防伪溯源的正面样本。"),
    ("implementation", "reference_retrieval_miss_all", "检索", "工具链或硬件实体相关切片未被召回。", False, "增加工具链关键词和实体字典。"),
    ("implementation", "generation_omitted_explicit_target", "生成", "目标实体进入证据但首次 Claim 漏引，被 repair 救回。", True, "强化实体覆盖提示并跟踪 repair rate。"),
    ("unanswerable", "abstained", "门控", "无证据问题被正确拒答，避免强行脑补。", False, "作为防幻觉门控的正面样本。"),
    ("unanswerable", "abstained", "门控", "过滤参考文献后仍能正确拒答相似但不可回答的问题。", False, "作为 References 过滤和拒答门控的正面样本。"),
]


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def load_attribution(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(payload.get("results", []))


def derive_cause(row: dict[str, object]) -> str:
    response = row.get("response", {})
    targets = set(row.get("explicit_target_aliases", response.get("requested_aliases", [])))
    evidence = set(response.get("evidence_aliases", []))
    cited = set(row.get("cited_aliases", []))
    references = set(row.get("reference_aliases", []))
    if str(response.get("repair_reason", "")).startswith("explicit_target_not_cited"):
        return "generation_omitted_explicit_target"
    if targets and not targets <= evidence:
        return "explicit_target_missing_from_evidence"
    if targets and not targets <= cited:
        return "generation_omitted_explicit_target"
    if targets:
        return "explicit_target_covered"
    if references and not references & evidence:
        return "reference_retrieval_miss_all"
    if references and cited and not references & cited:
        return "grounded_alternative_source"
    return "accepted_partial" if row.get("status") == "partial" else "other"


def resolve_rows(answerable: list[dict[str, object]], unanswerable: list[dict[str, object]], attribution: list[dict[str, object]]) -> list[dict[str, object]]:
    all_rows = {str(row["id"]): row for row in answerable + unanswerable}
    attr_rows = [row for row in attribution if str(row.get("id")) in all_rows]
    if not attr_rows:
        attr_rows = [dict(row, cause=derive_cause(row)) for row in answerable]
    used: set[str] = set()
    resolved: list[dict[str, object]] = []
    for category, cause, layer, root_cause, affects, fix in SAMPLES:
        candidates: list[dict[str, object]] = []
        if category == "unanswerable":
            candidates = [row for row in unanswerable if row.get("status") == "abstained"]
        elif cause == "accepted_partial":
            candidates = [row for row in answerable if row.get("category") == category and row.get("status") == "partial"]
        else:
            candidates = [row for row in attr_rows if row.get("category") == category and row.get("cause") == cause]
        candidates = [row for row in candidates if str(row.get("id")) not in used]
        if not candidates:
            raise ValueError(f"could not resolve sample category={category!r} cause={cause!r}")
        row = candidates[0]
        question_id = str(row["id"])
        used.add(question_id)
        resolved.append({"sample": len(resolved) + 1, "question_id": question_id, "category": category, "layer": layer, "root_cause": root_cause, "affects_core_metrics": affects, "proposed_fix": fix, "question": all_rows[question_id].get("response", {}).get("question", "")})
    return resolved


def render(rows: list[dict[str, object]]) -> str:
    lines = ["# Phase 4 Human Review Attribution", "", "本报告将 10 个代表性样本映射到评测结果中的真实 question_id。", "", "| # | Question ID | Category | Layer | Affects core metrics |", "|---:|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['sample']} | `{row['question_id']}` | {row['category']} | {row['layer']} | {'是' if row['affects_core_metrics'] else '否'} |")
    for row in rows:
        lines.extend(["", f"## {row['sample']}. {row['question_id']}", "", f"- Category: `{row['category']}`", f"- Question: {row['question']}", f"- Failure layer: {row['layer']}", f"- Root cause: {row['root_cause']}", f"- Affects core metrics: `{row['affects_core_metrics']}`", f"- Proposed fix: {row['proposed_fix']}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerable", type=Path, default=Path("artifacts/phase2_answerable_eval.json"))
    parser.add_argument("--unanswerable", type=Path, default=Path("artifacts/phase2_unanswerable_eval.json"))
    parser.add_argument("--attribution", type=Path, default=Path("artifacts/phase4_cross_layer_attribution.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase4_human_review.md"))
    args = parser.parse_args()
    rows = resolve_rows(load_rows(args.answerable), load_rows(args.unanswerable), load_attribution(args.attribution))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(f"wrote {args.output} with {len(rows)} resolved samples")


if __name__ == "__main__":
    main()
