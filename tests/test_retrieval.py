import unittest

from sc_rag.models import Chunk
from sc_rag.search import BM25Index
from sc_rag.text import expand_query


class RetrievalTests(unittest.TestCase):
    def test_chinese_domain_query_is_expanded(self) -> None:
        expanded = expand_query("随机计算中的相关性是什么？")
        self.assertIn("stochastic computing", expanded)
        self.assertIn("correlation", expanded)

    def test_search_returns_ranked_citable_chunks(self) -> None:
        chunks = [
            Chunk(chunk_id="a", paper_id="p1", alias="SCC", title="Correlation", page=3, text="stochastic cross-correlation measures correlation"),
            Chunk(chunk_id="b", paper_id="p2", alias="CNN", title="Network", page=8, text="convolutional neural network accelerator"),
        ]
        result = BM25Index(chunks).search("随机计算中的相关性", limit=1)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].chunk_id, "a")
        self.assertEqual(result.hits[0].page, 3)
        self.assertEqual(result.hits[0].rank, 1)
        self.assertEqual(result.hits[0].bm25_rank, 1)
