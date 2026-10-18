import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]


def read_jsonl(relative_path: str) -> list[dict[str, object]]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DataContractTests(unittest.TestCase):
    def test_manifest_has_exactly_50_unique_papers(self) -> None:
        papers = read_jsonl("data/metadata/papers.jsonl")
        self.assertEqual(len(papers), 50)
        self.assertEqual(len({paper["paper_id"] for paper in papers}), 50)
        self.assertEqual(len({paper["alias"] for paper in papers}), 50)
        self.assertEqual(len({paper["sha256"] for paper in papers}), 50)
        self.assertTrue(all(int(paper["pages"]) > 0 for paper in papers))
        self.assertTrue(all(int(paper["text_characters"]) > 100 for paper in papers))

    def test_benchmark_question_and_gold_contracts(self) -> None:
        questions = read_jsonl("data/eval/questions.jsonl")
        gold = read_jsonl("data/eval/gold_answers.jsonl")
        self.assertEqual(len(questions), 58)
        self.assertEqual(len(gold), 58)
        self.assertEqual(
            Counter(question["category"] for question in questions),
            {"fundamentals": 15, "single_paper": 15, "cross_paper": 10, "implementation": 10, "unanswerable": 8},
        )
        self.assertEqual([question["id"] for question in questions], [answer["id"] for answer in gold])
        self.assertTrue(all("answer" not in question for question in questions))

    def test_gold_aliases_resolve_to_manifest(self) -> None:
        aliases = {paper["alias"] for paper in read_jsonl("data/metadata/papers.jsonl")}
        gold = read_jsonl("data/eval/gold_answers.jsonl")
        self.assertTrue(all(set(answer["gold_aliases"]) <= aliases for answer in gold))

    def test_extraction_does_not_capture_document_notes(self) -> None:
        gold = read_jsonl("data/eval/gold_answers.jsonl")
        self.assertTrue(all("审核与冻结建议" not in str(answer["answer"]) for answer in gold))

