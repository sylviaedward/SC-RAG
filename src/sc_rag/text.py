from __future__ import annotations

import re
import unicodedata


TERM_EXPANSIONS = {
    "随机计算": "stochastic computing SC",
    "随机比特流": "stochastic bit stream bitstream",
    "随机数生成器": "stochastic number generator SNG random number generator",
    "单极": "unipolar",
    "双极": "bipolar",
    "相关性": "correlation stochastic cross-correlation SCC",
    "低差异": "low discrepancy sequence",
    "确定性": "deterministic bit stream",
    "乘法器": "multiplier multiplication",
    "精度": "accuracy error precision",
    "面积": "area hardware cost",
    "延迟": "latency delay cycle",
    "能耗": "energy power",
    "卷积神经网络": "convolutional neural network CNN",
    "容错": "fault tolerance error resilience",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def expand_query(query: str) -> str:
    additions = [english for chinese, english in TERM_EXPANSIONS.items() if chinese in query]
    return " ".join([query, *additions]).strip()


def tokenize(value: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value).lower()
    latin = re.findall(r"[a-z][a-z0-9_-]*|\d+(?:\.\d+)?", value)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", value)
    chinese = [run[index : index + 2] for run in chinese_runs for index in range(len(run) - 1)]
    return latin + chinese

