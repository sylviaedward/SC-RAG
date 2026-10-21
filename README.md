# SC-RAG

SC-RAG is an evidence-grounded research agent for stochastic-computing literature. It turns a 50-paper corpus into page-aware evidence, retrieves across Chinese and English terminology, validates every claim against trusted spans, and supports two-pass literature review with auditable comparisons and falsifiable research ideas.

The project is designed as a reproducible RAG system rather than a chat wrapper:

```text
PDF corpus → manifest → page-aware chunks → BM25 + dense RRF retrieval
          → alias-aware evidence planning → structured generation
          → citation/span validation → answer or explicit abstention
          → literature-review extraction → cross-paper synthesis
```

The public demo also exposes a bounded Agent control layer in the same
repository. It classifies the request, records a retrieval plan, performs
targeted alias supplementation when needed, applies the grounding gate, and
returns an `agent_trace` that the Streamlit UI can inspect. The model never
chooses arbitrary tools or receives authority to bypass citation validation.

## What makes it interview-worthy

- **Grounding contract:** the model can cite only application-issued span IDs; excerpts and PDF pages are resolved from trusted retrieval metadata.
- **Alias-aware planning:** explicit papers such as `[Sobol]` and `[P2LSG]` are fetched and covered independently, preventing one-paper dominance in comparison questions.
- **Fail-closed behavior:** malformed output, unknown spans, unsupported claims, incomplete comparisons, and insufficient evidence become an abstention rather than a plausible hallucination.
- **Two-pass research workflow:** each paper is first reduced to typed facts, then a synthesis pass compares only validated facts and binds every idea to evidence.
- **Reproducible evaluation:** answerable/unanswerable regression sets, retrieval metrics, citation-contract checks, latency, repair counts, and cross-layer failure attribution are saved as JSON.
- **Usable demo:** FastAPI exposes the contract; Streamlit presents answers, page evidence, comparisons, and research ideas.
- **Agent harness:** every answer records classification, plan, retrieval rounds, evidence gating, repair, and finalization steps for debugging and interviews.

## Run modes

SC-RAG supports three run modes. They use the same retrieval and answering code; choose the mode based on the audience and available resources.

| Mode | Use case | What it requires |
| --- | --- | --- |
| **Cloud development** | Build the corpus, run evaluations, inspect traces, and develop in Codespaces | Python environment, Poppler, corpus archives, and model credentials |
| **Local lightweight run** | Run the API and Streamlit directly on a laptop or workstation | Python 3.11+, Poppler, generated artifacts, and model credentials |
| **Containerized demo** | Let an interviewer start the complete API + UI with one command | Docker Compose, generated artifacts, and a model credential in `.env` |

Cloud development and containerized deployment are intentionally separate. Codespaces is useful for indexing and evaluation, while an interviewer can run the already-built demo locally with Docker. Docker is not required in the Codespace.

## Quick start: local development

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

The supplied archives may contain more papers than the benchmark's canonical
50-paper corpus. To import the original archives and select the exact papers
listed by the checked-in manifest in one step, use:

```bash
make prepare-corpus ARCHIVES="imports/SC_Liter*.zip"
make manifest
make ingest
```

This staging-and-selection flow avoids the `Expected 50 unique PDFs, found
100` error when the four source archives contain additional papers.

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

### Docker demo: one-command interview setup

Docker is the recommended interview path because it starts the FastAPI backend and Streamlit UI together with the same pinned application image. The interviewer needs Docker Desktop (or Docker Engine + Compose), a copy of the repository, the generated `artifacts/` directory, and one complete LLM variable group.

The public repository excludes raw PDFs and generated indexes. Prepare `artifacts/` once in Codespaces or copy a prepared artifact bundle next to the checkout. The artifact bundle should contain:

```text
artifacts/chunks.jsonl
artifacts/dense/embeddings.npy
artifacts/dense/metadata.json
```

Then run:

```bash
cp .env.example .env
# Fill OPENAI_*, DEEPSEEK_*, or SC_RAG_LLM_* in .env.
docker compose up --build -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Open `http://127.0.0.1:8501` for the UI and `http://127.0.0.1:8000/docs` for the API contract. The UI container calls the API over the Compose network, so no host-side `SC_RAG_API_URL` setting is needed.

Useful operations:

```bash
docker compose ps                 # service status
docker compose logs -f --tail=100 # troubleshoot startup
docker compose down               # stop and remove the demo
```

`/health` checks the indexed corpus. `/ready` also checks the dense index and complete LLM configuration. If the API port is occupied, choose alternate host ports without changing the containers:

```bash
SC_RAG_API_PORT=8001 SC_RAG_UI_PORT=8502 docker compose up --build -d
```

Then open port `8502` and use `http://127.0.0.1:8001` for the API. The first Docker build downloads the Python and ML dependencies and can take several minutes; later starts use the image cache. Docker mounts `artifacts/` and `data/` read-only, and `.env` is ignored by Git.

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
- `GET /ready` — full hybrid answering readiness, including dense artifacts and LLM configuration
- `GET /search?q=...` — ranked citable chunks
- `POST /answer` — grounded answer with claims, citations, status, and abstention reason
- `POST /literature-review` — fact-bound multi-paper review and research ideas

The public repository intentionally keeps implementation and product documentation concise. Detailed learning notes and interview preparation are maintained separately from the checkout.
