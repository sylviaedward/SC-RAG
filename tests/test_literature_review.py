import json
import unittest

from sc_rag.literature_review import LiteratureReviewService
from sc_rag.models import Chunk, ReviewDimension
from sc_rag.search import BM25Index


class SequenceClient:
    model_name = "fake-review-model"

    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return json.dumps(self.outputs[len(self.prompts) - 1])


def extraction(text: str) -> dict[str, object]:
    return {
        "method": [{"text": text, "span_ids": ["E1.S1"]}],
        "bitstream": [],
        "hardware": [],
        "evaluation": [],
        "limitations": [],
    }


class LiteratureReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index(
            [
                Chunk(
                    chunk_id="sobol:p1:c0",
                    paper_id="sobol",
                    alias="Sobol",
                    title="Sobol paper",
                    page=1,
                    text="Sobol sequences improve stochastic multiplication accuracy.",
                ),
                Chunk(
                    chunk_id="p2lsg:p2:c0",
                    paper_id="p2lsg",
                    alias="P2LSG",
                    title="P2LSG paper",
                    page=2,
                    text="P2LSG reduces sequence generator hardware cost.",
                ),
            ]
        )

    def test_two_pass_review_keeps_fact_and_source_boundaries(self) -> None:
        synthesis = {
            "comparison": [
                {
                    "dimension": "method",
                    "cells": [
                        {"alias": "Sobol", "summary": "Uses Sobol sequences.", "fact_ids": ["F1"]},
                        {"alias": "P2LSG", "summary": "Uses powers-of-two sequences.", "fact_ids": ["F2"]},
                    ],
                    "conflicts": [],
                }
            ],
            "ideas": [
                {
                    "observed_gap": "Accuracy and generator cost are evaluated separately.",
                    "evidence_fact_ids": ["F1", "F2"],
                    "inferred_limitation": "The joint trade-off is unknown.",
                    "hypothesis": "A hybrid generator can improve the Pareto frontier.",
                    "proposed_experiment": "Compare equal-length streams on matched circuits and technology.",
                    "falsifier": "No design improves both MSE and area over either baseline.",
                    "uncertainty": "The papers use different experimental conditions.",
                }
            ],
        }
        client = SequenceClient(
            [
                extraction("Sobol sequences are used for stochastic multiplication."),
                extraction("Powers-of-two sequences target lower generator cost."),
                synthesis,
            ]
        )

        result = LiteratureReviewService(self.index, client).review(
            "low-discrepancy bit-stream generators",
            ["Sobol", "P2LSG"],
            chunks_per_paper=1,
        )

        self.assertEqual(len(client.prompts), 3)
        self.assertEqual([paper.alias for paper in result.papers], ["Sobol", "P2LSG"])
        self.assertEqual(result.papers[0].method[0].fact_id, "F1")
        self.assertEqual(result.papers[1].method[0].fact_id, "F2")
        self.assertEqual(result.comparison[0].dimension, ReviewDimension.method)
        self.assertEqual(result.comparison[0].cells[1].citations[0].alias, "P2LSG")
        self.assertEqual(len(result.comparison), 5)
        self.assertEqual({item.alias for item in result.ideas[0].evidence}, {"Sobol", "P2LSG"})
        self.assertEqual(result.ideas[0].novelty_status, "unverified")
        self.assertIn("<VALIDATED_PAPER_FACTS>", client.prompts[-1])
        self.assertNotIn("<UNTRUSTED_PAPER_SPANS>", client.prompts[-1])

    def test_synthesis_cannot_invent_fact_ids(self) -> None:
        synthesis = {
            "comparison": [
                {
                    "dimension": "method",
                    "cells": [{"alias": "Sobol", "summary": "unsupported", "fact_ids": ["F99"]}],
                    "conflicts": [],
                }
            ],
            "ideas": [],
        }
        client = SequenceClient([extraction("fact a"), extraction("fact b"), synthesis])

        with self.assertRaisesRegex(ValueError, "phase3_unknown_fact:F99"):
            LiteratureReviewService(self.index, client).review("sequence generators", ["Sobol", "P2LSG"], 1)


if __name__ == "__main__":
    unittest.main()
