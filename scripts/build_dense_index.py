#!/usr/bin/env python3
"""Build persistent multilingual dense embeddings for Phase 1 retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_rag.dense import DEFAULT_EMBEDDING_MODEL, build_dense_artifacts, create_embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("artifacts/dense/embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/dense/metadata.json"))
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--provider", choices=("local", "api"), default="local")
    args = parser.parse_args()
    metadata = build_dense_artifacts(
        args.chunks,
        args.matrix,
        args.metadata,
        create_embedder(args.provider, args.model),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

