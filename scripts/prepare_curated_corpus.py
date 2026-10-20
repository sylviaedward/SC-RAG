#!/usr/bin/env python3
"""Import uploaded archives and keep the repository's canonical 50-paper corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from import_corpus_archives import import_archives


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(archives: list[Path], manifest_path: Path, output_dir: Path) -> int:
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    expected = {str(row["filename"]): str(row["sha256"]) for row in manifest}
    if len(expected) != 50:
        raise ValueError(f"Canonical manifest must contain 50 unique files, found {len(expected)}")

    with tempfile.TemporaryDirectory(prefix="sc-rag-import-") as staging:
        staging_dir = Path(staging)
        import_archives(archives, staging_dir)
        missing = sorted(filename for filename in expected if not (staging_dir / filename).is_file())
        if missing:
            raise FileNotFoundError(f"Archives do not contain {len(missing)} canonical PDFs: {missing[:3]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, expected_hash in expected.items():
            source = staging_dir / filename
            if sha256(source) != expected_hash:
                raise ValueError(f"SHA-256 mismatch for canonical PDF: {filename}")
            shutil.copy2(source, output_dir / filename)
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/papers.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/papers"))
    args = parser.parse_args()
    count = prepare(args.archives, args.manifest, args.output)
    print(f"prepared={count} canonical_pdfs={count} output={args.output}")


if __name__ == "__main__":
    main()
