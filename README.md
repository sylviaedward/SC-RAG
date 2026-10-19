# SC-RAG

An evidence-grounded literature assistant for stochastic computing (SC). It ingests 50 papers into page-aware chunks, runs bilingual hybrid retrieval, and produces claim-level answers whose quotations, chunks, and PDF pages are validated before delivery.

## Current capabilities

- Verified 50-paper corpus manifest with stable hashes and aliases
- 58-question benchmark split into question-only and isolated gold files
- Page-aware PDF ingestion with stable chunk IDs
- Chinese/English domain term expansion and BM25 retrieval
- Search API returning paper title, benchmark alias, PDF page, score, and source text
- Retrieval evaluation using gold paper aliases
- BM25 + multilingual dense retrieval with reciprocal-rank fusion
- OpenAI-compatible grounded generation with application-resolved evidence-span citations
- Explicit abstention for missing evidence or invalid model output

Raw PDFs, source evaluation documents, generated chunks, and evaluation reports are intentionally ignored by Git.

## Run in GitHub Codespaces

Start with the environment and tests; these do not require the raw papers:

```bash
make system-deps
make setup
make test
```

Create `imports/`, upload the four `SC_Liter*.zip` archives into it, then build the local index:

```bash
mkdir -p imports
make import-corpus ARCHIVES="imports/SC_Liter*.zip"
make manifest
make ingest
```

Run a query, the benchmark, and the API:

```bash
make search QUERY="随机计算中的相关性如何影响乘法器精度？"
make evaluate
make answer QUERY="单极格式下什么逻辑门实现乘法？" ANSWER_MODE=fundamentals
make api
```

If port 8000 is already occupied, first check whether SC-RAG is already running with `curl http://127.0.0.1:8000/health`. Otherwise start on another port with `make api PORT=8001`.

Python 3.11+ and Poppler (`pdfinfo`, `pdftotext`) are required. `make system-deps` installs Poppler through `apt` when it is missing. See the [Chinese workflow and code guide](docs/workflows.zh-CN.md) for expected outputs, the processing rationale, code locations, debugging exercises, and the readiness gate for each phase.

## MVP path

1. Freeze a domain-reviewed benchmark and atomic scoring rubric.
2. Add dense retrieval, reciprocal-rank fusion, and a reranker; compare every stage against this BM25 baseline.
3. Add OpenAI-compatible generation with claim-level citations and explicit abstention.
4. Add literature-comparison and research-gap workflows over selected evidence sets.
5. Add a Streamlit review UI, request tracing, Docker packaging, and a reproducible evaluation report.

See [the data audit](docs/data_audit.md) for known corpus and benchmark limitations.
The [completion and learning roadmap](docs/learning_roadmap.md) pairs every implementation phase with concepts, acceptance criteria, and likely interview questions.
For retrieval experiments, follow the [Phase 1 Chinese guide](docs/phase1_guide.zh-CN.md). For alias-aware evidence planning, grounded answers, citation validation, and abstention evaluation, follow the [Phase 2 Chinese guide](docs/phase2_guide.zh-CN.md).
