# Data audit

Audit date: 2026-10-18.

## Corpus

- Four uploaded archives contained 51 PDF files. One copy of `Understanding_variance_propagation_in_stochastic_computing_systems.pdf` was duplicated byte-for-byte.
- The cleaned corpus contains 50 unique papers, 645 PDF pages, and about 107 MB of source material.
- Every PDF is unencrypted and yields text without OCR. Layout and formula extraction still need page-level quality checks during evaluation.
- The 50 canonical aliases in `data/metadata/papers.jsonl` match the benchmark aliases. `[Bipolar-SNG?]` in the source document is an editorial placeholder, not a missing paper.
- Raw PDFs stay in `data/raw/papers/` and are ignored by Git. The tracked manifest records stable IDs, aliases, titles, paths, hashes, page counts, extracted character counts, and available DOIs.

## Benchmark

- The source PDF says 57 questions, but contains 58: A=15, B=15, C=10, D=10, E=8.
- Questions and gold answers are stored separately. Gold answers must never enter the retrieval index or generation context.
- All review checkboxes in the source are empty, so the benchmark remains a draft pending domain review.
- A4 mixes representational resolution (`2^k`) with the common stochastic sampling cost (`2^(2k)`). A7 uses simplified convergence wording. Numeric claims and page references in B/D should be checked against the papers before the benchmark is frozen.
- Compound questions need atomic scoring for answer correctness and citation correctness. The eight E questions should score abstention separately.
- Research ideation is intentionally outside the 58-question factual benchmark. It needs a qualitative rubric covering evidence traceability, novelty, feasibility, and stated uncertainty.

The extraction scripts enforce counts, matching IDs, unique corpus hashes, resolvable aliases, and isolation of editorial notes.
