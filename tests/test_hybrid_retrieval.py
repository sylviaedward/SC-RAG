import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np

from sc_rag.dense import DenseIndex, OpenAICompatibleEmbedder, build_dense_artifacts
from sc_rag.hybrid import HybridRetriever, apply_paper_diversity, reciprocal_rank_fusion
from sc_rag.models import Chunk, SearchHit
from sc_rag.search import BM25Index


class FakeEmbedder:
    model_name = "fake-embedding-model"

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        vectors = []
        for text in texts:
            if "correlation" in text.lower() or "相关" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [2.0 if "second" in passage.lower() else 1.0 for passage in passages]


def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", paper_id="p1", alias="P1", title="First", page=1, text="correlation metric"),
        Chunk(chunk_id="b", paper_id="p2", alias="P2", title="Second", page=2, text="second relevant method"),
        Chunk(chunk_id="c", paper_id="p1", alias="P1", title="First", page=3, text="correlation experiment"),
    ]


class DenseIndexTests(unittest.TestCase):
    def test_dense_artifacts_round_trip_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_path = root / "chunks.jsonl"
            chunks_path.write_text("".join(chunk.model_dump_json() + "\n" for chunk in chunks()), encoding="utf-8")
            matrix_path = root / "embeddings.npy"
            metadata_path = root / "metadata.json"

            metadata = build_dense_artifacts(chunks_path, matrix_path, metadata_path, FakeEmbedder())
            index = DenseIndex.from_artifacts(chunks_path, matrix_path, metadata_path, FakeEmbedder())
            result = index.search("相关性", limit=2)

            self.assertEqual(metadata["shape"], [3, 2])
            self.assertEqual([hit.chunk_id for hit in result.hits], ["a", "c"])
            self.assertTrue(all(hit.dense_rank is not None for hit in result.hits))

    @patch("urllib.request.urlopen")
    def test_openai_compatible_embeddings_are_ordered_and_normalized(self, urlopen: MagicMock) -> None:
        response = MagicMock()
        response.read.return_value = b'{"data":[{"index":1,"embedding":[0,2]},{"index":0,"embedding":[3,0]}]}'
        urlopen.return_value.__enter__.return_value = response
        embedder = OpenAICompatibleEmbedder("https://embedding.example/v1", "secret", "model", batch_size=2)

        vectors = embedder.encode(["first", "second"])

        np.testing.assert_allclose(vectors, np.asarray([[1, 0], [0, 1]], dtype=np.float32))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://embedding.example/v1/embeddings")


class HybridTests(unittest.TestCase):
    def test_rrf_preserves_component_ranks(self) -> None:
        first, second, _ = chunks()
        bm25 = [SearchHit(**first.model_dump(), score=9, rank=1)]
        dense = [SearchHit(**second.model_dump(), score=0.9, rank=1), SearchHit(**first.model_dump(), score=0.8, rank=2)]

        fused = reciprocal_rank_fusion(bm25, dense)

        self.assertEqual(fused[0].chunk_id, "a")
        self.assertEqual(fused[0].bm25_rank, 1)
        self.assertEqual(fused[0].dense_rank, 2)
        self.assertIsNotNone(fused[0].rrf_score)

    def test_diversity_limits_chunks_from_one_paper(self) -> None:
        hits = [SearchHit(**chunk.model_dump(), score=3 - index, rank=index + 1) for index, chunk in enumerate(chunks())]
        selected = apply_paper_diversity(hits, limit=2, max_per_paper=1)
        self.assertEqual([hit.paper_id for hit in selected], ["p1", "p2"])

    def test_reranker_changes_hybrid_order(self) -> None:
        corpus = chunks()
        dense = DenseIndex(corpus, np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32), FakeEmbedder())
        retriever = HybridRetriever(
            BM25Index(corpus),
            dense,
            reranker=FakeReranker(),
            candidate_k=3,
            rerank_k=3,
            max_per_paper=1,
        )
        result = retriever.search("相关性", limit=2)
        self.assertEqual(result.hits[0].paper_id, "p2")
        self.assertEqual(result.hits[0].reranker_score, 2.0)

