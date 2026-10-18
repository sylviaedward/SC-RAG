import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.import_corpus_archives import import_archives


class CorpusImportTests(unittest.TestCase):
    def test_flattens_safe_names_and_skips_content_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.zip"
            second = root / "second.zip"
            output = root / "papers"
            with zipfile.ZipFile(first, "w") as archive:
                archive.writestr("nested/paper.pdf", b"%PDF-first")
            with zipfile.ZipFile(second, "w") as archive:
                archive.writestr("../duplicate.pdf", b"%PDF-first")
                archive.writestr("other.pdf", b"%PDF-second")

            imported, duplicates = import_archives([first, second], output)

            self.assertEqual(imported, 2)
            self.assertEqual(duplicates, 1)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["other.pdf", "paper.pdf"])

