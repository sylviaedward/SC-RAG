# Architecture decisions for the research-agent demo

## Why this is an agent

The system has a bounded control loop rather than an unconstrained ReAct
chatbot. A request is classified, evidence is retrieved, explicit paper
aliases are supplemented, the evidence gate decides whether generation is
allowed, and the output is validated before it is returned. Comparison
questions can receive one deterministic query rewrite. Every transition is
returned in `agent_trace`.

This boundary is deliberate: a planner that makes an extra LLM call for every
question would add latency and another failure surface without improving the
grounding contract. The system still demonstrates planning, conditional
actions, stopping conditions, and observable state while keeping the critical
research behavior reproducible.

## Retrieval and reranking decision

The repository already contains BM25, dense retrieval, reciprocal-rank fusion,
and an optional cross-encoder reranker in `src/sc_rag/hybrid.py`. The default
answer path uses hybrid RRF when dense artifacts exist. Reranking remains an
explicit experiment because the Phase 1 measurements showed a large latency
cost and no consistent improvement in all-hit or MRR. A new model name is not
an improvement by itself; it must beat the existing baseline on the same 50
answerable questions and report latency.

RRF is retained because its rank-based combination is robust when BM25 and
dense scores have different scales. A weighted score would require calibrated
scores and a tuning split that the draft benchmark does not provide.

## Failure handling

- transient LLM gateway errors (408, 429, 5xx, and connection failures) use a
  bounded exponential retry controlled by `SC_RAG_LLM_RETRIES` and
  `SC_RAG_LLM_BACKOFF_SECONDS`;
- malformed JSON, unknown span IDs, unsupported claims, and incomplete
  comparisons fail closed;
- an empty retrieval result never enters generation;
- evidence is split into trusted span IDs under a character budget, so the
  model cannot enlarge its own context or invent citations;
- `/ready` reports whether the runtime is using `hybrid_rrf` or the explicit
  `bm25_fallback`.

## Resume-ready wording (fill only from measured reports)

- Built an evidence-grounded stochastic-computing research assistant over 50
  papers, combining page-aware chunking, bilingual BM25, dense RRF, and
  alias-aware evidence planning; achieved `{{HIT_ANY}}` retrieval hit-any and
  `{{CITATION_GOLD_HIT}}` citation hit-any on the 50-question benchmark
  (`artifacts/phase1_*.json` and Phase 2/4 reports).
- Designed a bounded Agent control loop with explicit classification,
  targeted supplementation, evidence gating, repair, and fail-closed
  citation validation; `agent_trace` makes every decision inspectable in the
  UI and API.
- Implemented two-pass literature review: typed per-paper facts first, then
  cross-paper synthesis constrained to validated fact IDs, with abstention on
  missing or conflicting evidence.
- Added reproducible exception handling for malformed model output and
  transient LLM failures; report retry counts and latency from the generated
  JSON rather than inventing performance claims.
