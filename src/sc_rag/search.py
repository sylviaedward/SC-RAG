from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Chunk, SearchHit, SearchResponse
from .text import expand_query, tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(f"{chunk.title} {chunk.alias} {chunk.text}") for chunk in chunks]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(chunks)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25Index":
        with path.open(encoding="utf-8") as handle:
            chunks = [Chunk.model_validate_json(line) for line in handle if line.strip()]
        return cls(chunks)

    def search(self, query: str, limit: int = 8) -> SearchResponse:
        expanded = expand_query(query)
        query_terms = Counter(tokenize(expanded))
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                term_score = self.idf.get(term, 0.0) * (frequency * (self.k1 + 1)) / (frequency + length_norm)
                score += term_score * (1 + 0.05 * math.log1p(query_frequency))
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))

        hits = [
            SearchHit(**self.chunks[item_index].model_dump(), score=round(score, 6), rank=rank)
            for rank, (score, item_index) in enumerate(scored[:limit], start=1)
        ]
        return SearchResponse(query=query, expanded_query=expanded, hits=hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local SC corpus")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    response = BM25Index.from_jsonl(args.index).search(args.query, args.limit)
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

