import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "attribute_phase2_failures.py"
SPEC = importlib.util.spec_from_file_location("attribute_phase2_failures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2AttributionTests(unittest.TestCase):
    def test_uses_actual_planned_evidence_for_explicit_targets(self) -> None:
        retrieval = {
            "retriever": "hybrid_rrf",
            "results": [
                {"id": "C1", "gold_aliases": ["A", "B"], "retrieved_aliases": ["A", "B"]},
                {"id": "C2", "gold_aliases": ["C", "D"], "retrieved_aliases": ["C", "D"]},
                {"id": "C3", "gold_aliases": ["E"], "retrieved_aliases": ["E"]},
            ],
        }
        answers = {
            "prompt_version": "v3",
            "results": [
                {
                    "id": "C1", "category": "cross_paper", "status": "partial", "cited_aliases": ["A"],
                    "explicit_target_aliases": ["A", "B"], "reference_aliases": ["A", "B"],
                    "response": {"evidence_aliases": ["A"]},
                },
                {
                    "id": "C2", "category": "cross_paper", "status": "partial", "cited_aliases": ["C"],
                    "explicit_target_aliases": ["C", "D"], "reference_aliases": ["C", "D"],
                    "response": {"evidence_aliases": ["C", "D"]},
                },
                {
                    "id": "C3", "category": "cross_paper", "status": "answered", "cited_aliases": ["E"],
                    "explicit_target_aliases": ["E"], "reference_aliases": ["E"],
                    "response": {"evidence_aliases": ["E"]},
                },
            ],
        }

        analysis = MODULE.attribute(retrieval, answers)

        self.assertEqual(analysis["cause_counts"]["explicit_target_missing_from_evidence"], 1)
        self.assertEqual(analysis["cause_counts"]["generation_omitted_explicit_target"], 1)
        self.assertEqual(analysis["cause_counts"]["explicit_target_covered"], 1)
        self.assertEqual(analysis["evidence_source"], "phase2_alias_planner")
        self.assertEqual(analysis["explicit_target_evidence_hit_all"], 0.6667)
        self.assertEqual(analysis["explicit_target_citation_hit_all"], 0.3333)


if __name__ == "__main__":
    unittest.main()
