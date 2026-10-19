import unittest

from sc_rag.ingest import split_page, strip_references_section


class IngestTests(unittest.TestCase):
    def test_removes_references_heading_and_following_entries(self) -> None:
        text = "实验结果\n误差为 2%。\n\nReferences\n[1] A paper.\n[2] Another paper."
        cleaned = strip_references_section(text)

        self.assertEqual(cleaned, "实验结果\n误差为 2%。")
        self.assertEqual(split_page(cleaned), ["实验结果\n误差为 2%。"])

    def test_does_not_remove_inline_references_word(self) -> None:
        text = "We compare references from two experiments."

        self.assertEqual(strip_references_section(text), text)


if __name__ == "__main__":
    unittest.main()
