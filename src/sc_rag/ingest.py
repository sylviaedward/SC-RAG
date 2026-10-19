from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

from .models import Chunk
from .text import normalize_text


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_page(text: str, target_chars: int = 1400, overlap_chars: int = 180) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) == 1:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]

    chunks: list[str] = []
    current = ""
    for part in paragraphs:
        if current and len(current) + len(part) + 1 > target_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = f"{overlap} {part}".strip()
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


_REFERENCES_HEADING = re.compile(
    r"^\s*(?:references|bibliography|参考文献|引用文献)\s*:?\s*$",
    re.IGNORECASE,
)


def strip_references_section(text: str) -> str:
    """Remove bibliography text so citations cannot become answer evidence."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _REFERENCES_HEADING.fullmatch(line):
            return "\n".join(lines[:index]).strip()
    return text


def iter_chunks(manifest_path: Path, repo_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paper in load_manifest(manifest_path):
        pdf_path = repo_root / str(paper["relative_path"])
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                page_text = strip_references_section(page.get_text("text"))
                for local_index, text in enumerate(split_page(page_text)):
                    identity = f"{paper['paper_id']}:{page_index}:{local_index}"
                    suffix = hashlib.sha1(identity.encode()).hexdigest()[:10]
                    chunks.append(
                        Chunk(
                            chunk_id=f"{paper['paper_id']}-{page_index}-{local_index}-{suffix}",
                            paper_id=str(paper["paper_id"]),
                            alias=str(paper["alias"]),
                            title=str(paper["title"]),
                            page=page_index,
                            text=text,
                        )
                    )
    return chunks


def write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            # ensure_ascii also escapes U+2028/U+2029 found in some publisher PDFs,
            # preserving the one-physical-line-per-record JSONL contract.
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build page-aware chunks from the SC corpus")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/papers.jsonl"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/chunks.jsonl"))
    args = parser.parse_args()
    chunks = iter_chunks(args.manifest, args.repo_root)
    write_jsonl(args.output, chunks)
    print(f"wrote {len(chunks)} page-aware chunks to {args.output}")


if __name__ == "__main__":
    main()

