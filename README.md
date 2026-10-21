# SC-RAG

SC-RAG is an evidence-grounded research agent for stochastic-computing (SC)
literature. It indexes a 50-paper corpus into page-aware evidence, validates
claims against retrieved spans, and supports two-pass literature review with
cross-paper synthesis.

```text
PDF → manifest → page-aware chunks → BM25 + dense RRF
    → Cross-Encoder rerank → paper diversity
    → alias-aware evidence planning
      ├─ targeted alias supplementation
      └─ comparison query rewrite
    → structured generation → citation/span validation
    → answer or abstention + agent trace
    → literature-review extraction → cross-paper synthesis
```

The API and Streamlit workspace use the same retrieval, evidence planning,
validation, and literature-review services.

## Design decisions

- **Grounding contract:** the model can cite only application-issued span IDs; excerpts and PDF pages are resolved from trusted retrieval metadata.
- **Deterministic multi-step retrieval:** missing aliases trigger targeted retrieval and comparison questions with insufficient candidates receive one query rewrite; the rule-driven policy keeps grounding predictable and records each step in `agent_trace`.
- **Cross-Encoder rerank:** the hybrid candidate set is reranked after bi-encoder retrieval; `rrf_score` breaks score ties, and `--no-reranker` disables this stage for ablation.
- **Fail-closed behavior:** malformed output, unknown spans, insufficient evidence, and incomplete comparisons produce an abstention.
- **Two-pass review:** each paper is reduced to typed facts before a synthesis pass combines only validated facts.
- **Reproducible evaluation:** the 58-question benchmark (50 answerable and 8 unanswerable) records retrieval metrics, citation-contract checks, latency, repair counts, and failure attribution as JSON.

## Run modes

SC-RAG supports three run modes. They use the same application services.

| Mode | Use case | Requirements |
| --- | --- | --- |
| **Cloud development** | Build the corpus, run evaluations, inspect traces, and develop in Codespaces | Python, Poppler, corpus archives, and model credentials |
| **Local lightweight run** | Run the API and Streamlit directly on a workstation | Python 3.11+, Poppler, generated artifacts, and model credentials |
| **Containerized demo** | Run the API and UI together with Docker Compose | Docker Compose, generated artifacts, and one LLM variable group in `.env` |

Cloud development and containerized deployment use the same source code but
have separate dependency and artifact preparation steps.

## Quick start: local development

Python 3.11+ and Poppler are required. In a new environment:

```bash
make system-deps
make setup
make test
```

To build the local corpus index, place the supplied `SC_Liter*.zip` archives
under `imports/`:

```bash
mkdir -p imports
make import-corpus ARCHIVES="imports/SC_Liter*.zip"
make manifest
make ingest
make dense-index
```

If the archives contain additional papers, select the canonical 50-paper
manifest during preparation:

```bash
make prepare-corpus ARCHIVES="imports/SC_Liter*.zip"
make manifest
make ingest
```

Configure one complete OpenAI-compatible LLM variable group:

```text
DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY + DEEPSEEK_MODEL_ID
```

`OPENAI_*` and `SC_RAG_LLM_*` groups are also supported. Credentials are read
from the environment and are not committed.

Run the API and UI in separate terminals:

```bash
make api PORT=8000
make setup-ui
make ui
```

Open the forwarded Streamlit port `8501`. A command-line request can be sent
with:

```bash
make answer QUERY="单极格式下什么逻辑门实现乘法？" ANSWER_MODE=fundamentals
```

The Streamlit workspace displays the answer, claim-level citations with paper
alias, PDF page, and excerpt, and an expandable agent trace.

![Streamlit workspace](docs/ui.png)

### Containerized demo

To run the demo, you need Docker Desktop (or Docker Engine with Compose), a
copy of the repository, the generated `artifacts/` directory, and one complete
LLM variable group.

The artifact directory should contain:

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

Open `http://127.0.0.1:8501` for the UI and
`http://127.0.0.1:8000/docs` for the API schema. The UI container reaches the
API through the Compose network.

Useful operations:

```bash
docker compose ps
docker compose logs -f --tail=100
docker compose down
```

If the host ports are occupied, choose different host ports:

```bash
SC_RAG_API_PORT=8001 SC_RAG_UI_PORT=8502 docker compose up --build -d
```

The application container uses `hybrid_rrf` when dense artifacts are present.
When they are unavailable, `/ready` reports `bm25_fallback`.

## Evaluation

The benchmark contains 50 answerable and 8 unanswerable questions. Run the
complete answer evaluation with:

```bash
make evaluate-phase4
```

For the reranker ablation, run:

```bash
python scripts/run_reranker_ablation.py
```

The script compares `no_reranker` and `reranker` on the same 58 questions and
writes `reports/reranker_ablation.json`. It reports Top-5 recall, answerable
accuracy, abstain correctness, and retrieval/end-to-end p50 and p95 latency.
Run the script to fill the measured values; the placeholders below are not
performance claims.

| Metric | no_reranker | reranker |
| --- | ---: | ---: |
| Top-5 recall | {{X}} | {{X}} |
| Answerable accuracy | {{X}} | {{X}} |
| Abstain correctness | {{X}} | {{X}} |
| Retrieval p50 latency (ms) | {{X}} | {{X}} |
| Retrieval p95 latency (ms) | {{X}} | {{X}} |
| End-to-end p50 latency (ms) | {{X}} | {{X}} |
| End-to-end p95 latency (ms) | {{X}} | {{X}} |

To inspect the ten human-reviewed attribution cases:

```bash
python scripts/generate_phase4_human_review.py
```

Generated indexes, traces, reports, raw PDFs, and source benchmark documents
are excluded from Git.

## Repository map

```text
src/sc_rag/answering.py          grounded answering and citation validation
src/sc_rag/ingest.py             PDF extraction and References filtering
src/sc_rag/search.py             BM25 retrieval and bilingual query expansion
src/sc_rag/dense.py              dense index and embedding providers
src/sc_rag/hybrid.py             BM25/dense RRF, reranking, and paper diversity
src/sc_rag/models.py             API, citation, review, and agent-trace schemas
src/sc_rag/literature_review.py  two-pass paper extraction and synthesis
src/sc_rag/api.py                FastAPI contract
ui/app.py                        Streamlit research workspace
scripts/evaluate_phase4.py       reproducible answer evaluation and traces
scripts/run_reranker_ablation.py reranker comparison on the 58-question set
tests/                           retrieval, grounding, gating, and schema tests
```

## API contract

- `GET /health` — indexed corpus readiness
- `GET /ready` — answering readiness, model configuration, and active retriever
- `GET /search?q=...` — ranked citable chunks
- `POST /answer` — grounded answer with claims, citations, status, abstention reason, and optional `agent_trace` containing `steps`, `retrieval_rounds`, and `stop_reason`
- `POST /literature-review` — fact-bound multi-paper review and research ideas

Extended design notes are maintained outside this repository.
