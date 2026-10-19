import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_phase2_failures.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase2_failures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2AnalysisTests(unittest.TestCase):
    def test_separates_model_abstention_from_quote_rejection(self) -> None:
        report = {
            "results": [
                {
                    "id": "A1",
                    "category": "fundamentals",
                    "status": "answered",
                    "explicit_target_aliases": ["A"],
                    "explicit_target_citation_hit_all": True,
                    "reference_source_hit_any": True,
                    "reference_source_recall": 1.0,
                    "gold_hit_any": True,
                    "gold_hit_all": True,
                    "response": {"reason": "supported", "evidence_sufficiency": "sufficient"},
                },
                {
                    "id": "A2",
                    "category": "fundamentals",
                    "status": "abstained",
                    "gold_hit_any": False,
                    "gold_hit_all": False,
                    "response": {
                        "reason": "citation_quote_not_found_in_evidence",
                        "evidence_sufficiency": "insufficient",
                    },
                },
                {
                    "id": "A3",
                    "category": "cross_paper",
                    "status": "abstained",
                    "gold_hit_any": False,
                    "gold_hit_all": False,
                    "response": {"reason": "Evidence does not cover every requested fact.", "evidence_sufficiency": "insufficient"},
                },
            ]
        }

        analysis = MODULE.analyze(report)

        self.assertEqual(analysis["conditional_reference_source_hit_any"], 1.0)
        self.assertEqual(analysis["conditional_mean_reference_source_recall"], 1.0)
        self.assertEqual(analysis["reason_counts"]["citation_quote_not_found_in_evidence"], 1)
        self.assertEqual(analysis["reason_counts"]["model_declared_insufficient"], 1)

    def test_partial_answers_count_as_accepted(self) -> None:
        report = {
            "results": [
                {
                    "id": "C1",
                    "category": "cross_paper",
                    "status": "partial",
                    "explicit_target_aliases": ["A", "B"],
                    "explicit_target_citation_hit_all": False,
                    "reference_source_hit_any": True,
                    "reference_source_recall": 0.5,
                    "gold_hit_any": True,
                    "gold_hit_all": False,
                    "response": {"reason": "some dimensions missing", "evidence_sufficiency": "partial"},
                }
            ]
        }

        analysis = MODULE.analyze(report)

        self.assertEqual(analysis["accepted_answers"], 1)
        self.assertEqual(analysis["partial_answers"], 1)
        self.assertEqual(analysis["conditional_reference_source_hit_any"], 1.0)
        self.assertEqual(analysis["conditional_mean_reference_source_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
