import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "attribute_phase2_failures.py"
SPEC = importlib.util.spec_from_file_location("attribute_phase2_failures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2AttributionTests(unittest.TestCase):
    def test_distinguishes_retrieval_loss_from_generation_omission(self) -> None:
        retrieval = {
            "retriever": "hybrid_rrf",
            "results": [
                {"id": "C1", "gold_aliases": ["A", "B"], "retrieved_aliases": ["A", "X"]},
                {"id": "C2", "gold_aliases": ["C", "D"], "retrieved_aliases": ["C", "D"]},
                {"id": "C3", "gold_aliases": ["E"], "retrieved_aliases": ["E"]},
            ],
        }
        answers = {
            "prompt_version": "v3",
            "results": [
                {"id": "C1", "category": "cross_paper", "status": "partial", "cited_aliases": ["A"]},
                {"id": "C2", "category": "cross_paper", "status": "partial", "cited_aliases": ["C"]},
                {"id": "C3", "category": "cross_paper", "status": "answered", "cited_aliases": ["E"]},
            ],
        }

        analysis = MODULE.attribute(retrieval, answers)

        self.assertEqual(analysis["cause_counts"]["retrieval_incomplete"], 1)
        self.assertEqual(analysis["cause_counts"]["generation_omitted_retrieved_gold"], 1)
        self.assertEqual(analysis["cause_counts"]["full_coverage"], 1)
        self.assertEqual(analysis["retrieval_hit_all"], 0.6667)
        self.assertEqual(analysis["citation_hit_all"], 0.3333)


if __name__ == "__main__":
    unittest.main()
