import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_phase2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_phase2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2EvaluationTests(unittest.TestCase):
    def test_separates_explicit_targets_from_reference_sources(self) -> None:
        rows = [
            {
                "expected_answerable": True,
                "status": "partial",
                "behavior_correct": True,
                "explicit_target_aliases": ["A", "B"],
                "explicit_target_evidence_hit_all": True,
                "explicit_target_citation_hit_all": False,
                "explicit_target_citation_recall": 0.5,
                "reference_source_hit_any": True,
                "reference_source_recall": 0.25,
                "legacy_reference_source_hit_all": False,
                "response": {
                    "claims": [{"text": "supported"}],
                    "generation_attempts": 2,
                    "repair_reason": "explicit_target_not_cited:B",
                },
            },
            {
                "expected_answerable": True,
                "status": "answered",
                "behavior_correct": True,
                "explicit_target_aliases": [],
                "explicit_target_evidence_hit_all": None,
                "explicit_target_citation_hit_all": None,
                "explicit_target_citation_recall": None,
                "reference_source_hit_any": True,
                "reference_source_recall": 1.0,
                "legacy_reference_source_hit_all": True,
                "response": {"claims": [{"text": "supported"}]},
            },
        ]

        report = MODULE.build_report("answerable", "test", rows, [10.0, 20.0])

        self.assertEqual(report["explicit_target_question_count"], 1)
        self.assertEqual(report["explicit_target_evidence_hit_all"], 1.0)
        self.assertEqual(report["explicit_target_citation_hit_all"], 0.0)
        self.assertEqual(report["mean_explicit_target_citation_recall"], 0.5)
        self.assertEqual(report["mean_reference_source_recall"], 0.625)
        self.assertEqual(report["legacy_reference_source_hit_all"], 0.5)
        self.assertEqual(report["generation_repair_rate"], 0.5)
        self.assertEqual(report["generation_repair_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
