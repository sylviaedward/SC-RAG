# SC-RAG

An evidence-grounded literature assistant for stochastic computing (SC). The first working slice ingests 50 papers into page-aware chunks, expands Chinese domain queries into English SC terminology, retrieves evidence with a reproducible BM25 baseline, and exposes citations through a FastAPI endpoint.

## Current capabilities

- Verified 50-paper corpus manifest with stable hashes and aliases
- 58-question benchmark split into question-only and isolated gold files
- Page-aware PDF ingestion with stable chunk IDs
- Chinese/English domain term expansion and BM25 retrieval
- Search API returning paper title, benchmark alias, PDF page, score, and source text
- Retrieval evaluation using gold paper aliases

Raw PDFs, source evaluation documents, generated chunks, and evaluation reports are intentionally ignored by Git.

## Setup

Python 3.11+ and Poppler (`pdfinfo`, `pdftotext`) are required. In this cloud environment, reuse the preinstalled scientific/API packages:

```bash
cd /workspace/SC-RAG
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

Place the 50 unique PDFs under `data/raw/papers/`, then build and validate the data:

```bash
.venv/bin/python scripts/build_corpus_manifest.py
.venv/bin/python scripts/extract_eval_benchmark.py data/eval/source/SC_RAG知识库评测题库_题目卷与答案卷.pdf
.venv/bin/sc-rag-ingest
.venv/bin/python -m unittest discover -s tests -v
```

Search from the command line:

```bash
.venv/bin/sc-rag-search '随机计算中的相关性如何影响乘法器精度？' --limit 5
```

Start the API:

```bash
.venv/bin/python -m uvicorn sc_rag.api:app --host 0.0.0.0 --port 8000
```

Check `GET /health` and query `GET /search?q=...&limit=5`.

Run the baseline evaluation:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --k 10
```

## MVP path

1. Freeze a domain-reviewed benchmark and atomic scoring rubric.
2. Add dense retrieval, reciprocal-rank fusion, and a reranker; compare every stage against this BM25 baseline.
3. Add OpenAI-compatible generation with claim-level citations and explicit abstention.
4. Add literature-comparison and research-gap workflows over selected evidence sets.
5. Add a Streamlit review UI, request tracing, Docker packaging, and a reproducible evaluation report.

See [the data audit](docs/data_audit.md) for known corpus and benchmark limitations.
The [completion and learning roadmap](docs/learning_roadmap.md) pairs every implementation phase with concepts, acceptance criteria, and likely interview questions.
