from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .models import Chunk, SearchHit, SearchResponse


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install Phase 1 dependencies with `make setup-hybrid`.") from error
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAICompatibleEmbedder:
    """Embedding client for an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, base_url: str, api_key: str, model_name: str, batch_size: int = 32) -> None:
        if not base_url or not api_key or not model_name:
            raise ValueError("Embedding base URL, API key, and model name are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size

    @classmethod
    def from_env(cls) -> "OpenAICompatibleEmbedder":
        return cls(
            base_url=os.getenv("EMBED_BASE_URL", ""),
            api_key=os.getenv("EMBED_API_KEY", ""),
            model_name=os.getenv("EMBED_MODEL_NAME", ""),
        )

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            payload = json.dumps({"model": self.model_name, "input": batch}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Embedding API returned HTTP {error.code}: {detail}") from error
            except urllib.error.URLError as error:
                raise RuntimeError(f"Could not reach the embedding API: {error.reason}") from error
            ordered = sorted(body["data"], key=lambda item: item["index"])
            if len(ordered) != len(batch):
                raise ValueError("Embedding API response count does not match request count")
            vectors.extend(item["embedding"] for item in ordered)
            if show_progress:
                print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)}")
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding API returned a zero vector")
        return matrix / norms


def create_embedder(provider: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    if provider == "local":
        return SentenceTransformerEmbedder(model_name)
    if provider == "api":
        return OpenAICompatibleEmbedder.from_env()
    raise ValueError(f"Unknown embedding provider: {provider}")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


def chunks_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passage_text(chunk: Chunk) -> str:
    return f"passage: {chunk.title}. {chunk.text}"


def build_dense_artifacts(
    chunks_path: Path,
    matrix_path: Path,
    metadata_path: Path,
    embedder: Embedder,
) -> dict[str, object]:
    chunks = read_chunks(chunks_path)
    vectors = embedder.encode([passage_text(chunk) for chunk in chunks], show_progress=True)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(f"Embedding shape {vectors.shape} does not match {len(chunks)} chunks")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, vectors.astype(np.float32))
    metadata = {
        "model_name": embedder.model_name,
        "chunks_sha256": chunks_sha256(chunks_path),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "shape": list(vectors.shape),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


class DenseIndex:
    name = "dense_multilingual_e5"

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Dense matrix row count must match chunk count")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.embedder = embedder

    @classmethod
    def from_artifacts(
        cls,
        chunks_path: Path,
        matrix_path: Path,
        metadata_path: Path,
        embedder: Embedder,
    ) -> "DenseIndex":
        chunks = read_chunks(chunks_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["model_name"] != embedder.model_name:
            raise ValueError("Embedding model differs from the model used to build the dense index")
        if metadata["chunks_sha256"] != chunks_sha256(chunks_path):
            raise ValueError("Chunks changed after the dense index was built; rebuild it")
        if metadata["chunk_ids"] != [chunk.chunk_id for chunk in chunks]:
            raise ValueError("Dense metadata chunk order does not match chunks")
        return cls(chunks, np.load(matrix_path), embedder)

    def search(self, query: str, limit: int = 8) -> SearchResponse:
        query_vector = self.embedder.encode([f"query: {query}"])[0]
        scores = self.vectors @ query_vector
        limit = min(limit, len(self.chunks))
        indices = np.argpartition(-scores, limit - 1)[:limit]
        indices = sorted(indices.tolist(), key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id))
        hits = [
            SearchHit(
                **self.chunks[index].model_dump(),
                score=round(float(scores[index]), 6),
                rank=rank,
                dense_rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        ]
        return SearchResponse(query=query, expanded_query=query, hits=hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense semantic search over the SC corpus")
    parser.add_argument("query")
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("artifacts/dense/embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/dense/metadata.json"))
    parser.add_argument("--provider", choices=("local", "api"), default="local")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    embedder = create_embedder(args.provider, args.model)
    response = DenseIndex.from_artifacts(args.chunks, args.matrix, args.metadata, embedder).search(
        args.query, args.limit
    )
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

