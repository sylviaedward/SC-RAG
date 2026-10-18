#!/usr/bin/env python3
"""Import PDFs from uploaded corpus ZIP files without unsafe path extraction."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path, PurePosixPath


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_archives(archives: list[Path], output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Finder stores resource forks as __MACOSX/._name.pdf. They share the PDF
    # suffix but are AppleDouble metadata rather than readable papers. Remove
    # any such files left by an earlier import before counting the corpus.
    for sidecar in output_dir.glob("._*.pdf"):
        sidecar.unlink()

    known_hashes: dict[str, Path] = {}
    for existing in output_dir.glob("*.pdf"):
        known_hashes[digest(existing.read_bytes())] = existing

    imported = 0
    duplicates = 0
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(f"Archive not found: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".pdf"):
                    continue
                member_path = PurePosixPath(member.filename)
                filename = member_path.name
                if "__MACOSX" in member_path.parts or filename.startswith("._"):
                    continue
                if not filename:
                    continue
                data = bundle.read(member)
                file_hash = digest(data)
                if file_hash in known_hashes:
                    duplicates += 1
                    continue
                destination = output_dir / filename
                if destination.exists():
                    raise ValueError(f"Different PDFs use the same filename: {filename}")
                destination.write_bytes(data)
                known_hashes[file_hash] = destination
                imported += 1
    return imported, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and deduplicate SC corpus archives")
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw/papers"))
    args = parser.parse_args()
    imported, duplicates = import_archives(args.archives, args.output)
    total = len(list(args.output.glob("*.pdf")))
    print(f"imported={imported} duplicate_files_skipped={duplicates} unique_pdfs={total}")
    if total != 50:
        raise SystemExit(f"Expected 50 unique PDFs, found {total}")


if __name__ == "__main__":
    main()

