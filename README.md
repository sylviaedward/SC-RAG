# SC-RAG

SC-RAG is an evidence-grounded research agent for stochastic-computing literature. It turns a 50-paper corpus into page-aware evidence, retrieves across Chinese and English terminology, validates every claim against trusted spans, and supports two-pass literature review with auditable comparisons and falsifiable research ideas.

The project is designed as a reproducible RAG system rather than a chat wrapper:

```text
PDF corpus → manifest → page-aware chunks → BM25 + dense RRF retrieval
          → alias-aware evidence planning → structured generation
          → citation/span validation → answer or explicit abstention
          → literature-review extraction → cross-paper synthesis
```

## What makes it interview-worthy

- **Grounding contract:** the model can cite only application-issued span IDs; excerpts and PDF pages are resolved from trusted retrieval metadata.
- **Alias-aware planning:** explicit papers such as `[Sobol]` and `[P2LSG]` are fetched and covered independently, preventing one-paper dominance in comparison questions.
- **Fail-closed behavior:** malformed output, unknown spans, unsupported claims, incomplete comparisons, and insufficient evidence become an abstention rather than a plausible hallucination.
- **Two-pass research workflow:** each paper is first reduced to typed facts, then a synthesis pass compares only validated facts and binds every idea to evidence.
- **Reproducible evaluation:** answerable/unanswerable regression sets, retrieval metrics, citation-contract checks, latency, repair counts, and cross-layer failure attribution are saved as JSON.
- **Usable demo:** FastAPI exposes the contract; Streamlit presents answers, page evidence, comparisons, and research ideas.

## Quick start

Python 3.11+ and Poppler are required. In a fresh Codespace:

```bash
make system-deps
make setup
make test
```

To build the local corpus index, place the supplied `SC_Liter*.zip` archives under `imports/`:

```bash
mkdir -p imports
make import-corpus ARCHIVES="imports/SC_Liter*.zip"
make manifest
make ingest
make dense-index
```

Configure one complete OpenAI-compatible LLM variable group in the environment:

```text
DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY + DEEPSEEK_MODEL_ID
```

or the equivalent `OPENAI_*` / `SC_RAG_LLM_*` group. Credentials are never committed or shown by the application.

Run the API and UI in separate terminals:

```bash
make api PORT=8000
make setup-ui
make ui
```

Open the forwarded port `8501` for the Streamlit workspace. For a command-line smoke test:

```bash
make answer QUERY="单极格式下什么逻辑门实现乘法？" ANSWER_MODE=fundamentals
```

## Evaluation

The benchmark contains 50 answerable and 8 unanswerable questions. Run the complete Phase 4 evaluation with one command:

```bash
make evaluate-phase4
```

The report records the dataset/configuration version, category metrics, latency, repair attempts, retrieval evidence aliases, model status, and failure taxonomy. To inspect the ten human-reviewed attribution cases:

```bash
python scripts/generate_phase4_human_review.py
```

Generated indexes, traces, reports, raw PDFs, and source benchmark documents stay outside Git through `.gitignore`.

## Repository map

```text
src/sc_rag/answering.py          grounded answering and citation validation
src/sc_rag/ingest.py             PDF extraction and References filtering
src/sc_rag/hybrid.py             BM25/dense reciprocal-rank fusion
src/sc_rag/literature_review.py  two-pass paper extraction and synthesis
src/sc_rag/api.py                FastAPI contract
ui/app.py                        Streamlit research workspace
scripts/evaluate_phase4.py       reproducible evaluation and traces
tests/                           retrieval, grounding, gating, and schema tests
```

## API contract

- `GET /health` — index readiness
- `GET /search?q=...` — ranked citable chunks
- `POST /answer` — grounded answer with claims, citations, status, and abstention reason
- `POST /literature-review` — fact-bound multi-paper review and research ideas

The public repository intentionally keeps implementation and product documentation concise. Detailed learning notes and interview preparation are maintained separately from the checkout.
