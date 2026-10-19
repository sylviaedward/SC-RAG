import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "assess_phase2_readiness.py"
SPEC = importlib.util.spec_from_file_location("assess_phase2_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def citation(alias: str) -> dict[str, object]:
    return {
        "alias": alias,
        "span_id": "E1.S1",
        "chunk_id": f"{alias}:p1:c0",
        "excerpt": "supporting source text",
    }


class Phase2ReadinessTests(unittest.TestCase):
    def test_alternative_reference_sources_are_diagnostic_not_gate(self) -> None:
        answerable = {
            "prompt_version": "v5",
            "model": "test",
            "reference_source_hit_any": 0.5,
            "mean_reference_source_recall": 0.5,
            "results": [
                {
                    "id": "A1",
                    "expected_answerable": True,
                    "status": "answered",
                    "explicit_target_aliases": [],
                    "reference_source_hit_any": False,
                    "response": {"claims": [{"text": "supported", "citations": [citation("Alternative")]}]},
                },
                {
                    "id": "B1",
                    "expected_answerable": True,
                    "status": "answered",
                    "explicit_target_aliases": ["Named"],
                    "explicit_target_evidence_hit_all": True,
                    "explicit_target_citation_hit_all": True,
                    "reference_source_hit_any": True,
                    "response": {"claims": [{"text": "supported", "citations": [citation("Named")]}]},
                },
            ],
        }
        unanswerable = {
            "prompt_version": "v5",
            "model": "test",
            "results": [
                {
                    "id": "E1",
                    "expected_answerable": False,
                    "status": "abstained",
                    "response": {"claims": []},
                }
            ],
        }

        result = MODULE.assess(
            answerable,
            unanswerable,
            expected_answerable=2,
            expected_unanswerable=1,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["diagnostics_not_gates"]["grounded_alternative_source_count"], 1)
        self.assertEqual(result["diagnostics_not_gates"]["grounded_alternative_source_ids"], ["A1"])

    def test_incomplete_benchmark_cannot_pass(self) -> None:
        result = MODULE.assess(
            {"results": [], "prompt_version": "v5", "model": "test"},
            {"results": [], "prompt_version": "v5", "model": "test"},
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["complete_answerable_set"])
        self.assertFalse(result["checks"]["complete_unanswerable_set"])


if __name__ == "__main__":
    unittest.main()
