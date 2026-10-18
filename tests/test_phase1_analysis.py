import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_phase1_failures.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase1_failures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase1AnalysisTests(unittest.TestCase):
    def test_reports_ranking_and_coverage_tradeoffs(self) -> None:
        base = {
            "retriever": "hybrid_rrf",
            "results": [
                {
                    "id": "Q1",
                    "category": "multi-paper",
                    "gold_aliases": ["A", "B"],
                    "retrieved_aliases": ["A", "B", "C"],
                    "any_hit": True,
                    "all_hit": True,
                },
                {
                    "id": "Q2",
                    "category": "single-paper",
                    "gold_aliases": ["D"],
                    "retrieved_aliases": ["C", "D"],
                    "any_hit": True,
                    "all_hit": True,
                },
            ],
        }
        reranked = {
            "retriever": "hybrid_rrf_reranked",
            "results": [
                {
                    "id": "Q1",
                    "category": "multi-paper",
                    "gold_aliases": ["A", "B"],
                    "retrieved_aliases": ["A", "C"],
                    "any_hit": True,
                    "all_hit": False,
                },
                {
                    "id": "Q2",
                    "category": "single-paper",
                    "gold_aliases": ["D"],
                    "retrieved_aliases": ["D", "C"],
                    "any_hit": True,
                    "all_hit": True,
                },
            ],
        }

        analysis = MODULE.compare_reports(base, reranked)

        self.assertEqual(analysis["coverage_harmed"], 1)
        self.assertEqual(analysis["rr_improved"], 1)
        self.assertEqual(analysis["all_hit_lost"], 1)
        self.assertEqual(analysis["any_hit_lost"], 0)


if __name__ == "__main__":
    unittest.main()
