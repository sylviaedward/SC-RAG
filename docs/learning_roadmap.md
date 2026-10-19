# SC-RAG completion and learning roadmap

The goal is a research assistant that can answer SC fundamentals, compare papers and implementations, identify evidence-backed gaps, cite exact sources, and abstain when the corpus does not support a claim. Each phase ends with code, measurements, and an interview-ready explanation.

## Working method

Use the same loop for every phase:

1. **Build:** implement the smallest measurable change.
2. **Read:** trace the request from entry point to output and explain each module without an assistant.
3. **Measure:** run the fixed benchmark and retain the before/after result.
4. **Diagnose:** inspect at least five failures and classify their causes.
5. **Explain:** write a short design note and answer the phase questions aloud.

Do not memorize every line. You should be able to change the code, predict its effect, locate failures, and justify the design.

## Phase 0 — Data and lexical baseline (completed)

Deliverables:

- 50-paper deduplicated manifest with hashes and page counts
- 58-question benchmark with question/gold isolation
- page-aware ingestion and stable chunk IDs
- bilingual term expansion, BM25 retrieval, CLI and FastAPI search
- reproducible retrieval metrics and data-contract tests

Current Top-10 baseline: Hit-any 0.86, Hit-all 0.58, MRR 0.7569.

You should understand:

- why raw PDFs and gold answers must not enter the same indexing path
- how chunk size and overlap affect recall, context continuity, and noise
- BM25 term frequency, inverse document frequency, and document-length normalization
- why Chinese questions over English papers need query translation or expansion
- why Hit@K and MRR measure different retrieval behavior

Likely interview questions:

- Why did you choose page-aware chunks instead of fixed token windows?
- Why can BM25 outperform embeddings for paper names, formulas, and rare hardware terms?
- How did you detect duplicate papers and prevent evaluation leakage?
- Why is an 86% Hit-any score insufficient for cross-paper questions?
- What is the difference between PDF page number and printed paper page number?

## Phase 1 — Hybrid retrieval (completed)

Build:

- multilingual dense embeddings and a persistent vector index
- BM25 and dense candidate retrieval
- reciprocal-rank fusion (RRF)
- cross-encoder or LLM reranking over the fused candidates
- alias/paper diversity control so one paper cannot occupy every result

Acceptance criteria:

- compare BM25, dense, hybrid, and hybrid+reranker on the same 50 answerable questions
- improve Hit-all@10 materially over 0.58
- record latency, index size, model name, configuration, and failure cases

You should understand cosine similarity, embedding model selection, RRF, reranking cost, recall/precision trade-offs, and why retrieval stages use different K values.

Likely interview questions:

- Why use RRF rather than directly adding BM25 and cosine scores?
- What happens when dense and sparse retrieval return the same chunks?
- Why rerank only a candidate set instead of the entire corpus?
- How would you update or delete one paper from the index?

## Phase 2 — Evidence-grounded answering (current)

Build:

- an OpenAI-compatible model adapter for the user's relay or DeepSeek
- answer modes for fundamentals, single-paper lookup, and cross-paper comparison
- structured output containing answer, claim-level citations, confidence, and evidence sufficiency
- explicit abstention when evidence is missing or contradictory
- context-budget controls and prompt-injection-resistant document wrapping
- alias-aware evidence planning for questions that explicitly name one or more papers

Acceptance criteria:

- every material factual claim maps to a retrieved chunk and PDF page
- the eight unanswerable questions are evaluated separately
- unsupported citations and malformed model output fail closed
- API keys remain environment variables and never enter Git or logs
- explicit paper targets and optional reference sources are scored separately; reference-source Hit-all is diagnostic only
- a single readiness command gates Phase 3 on the complete 50-answerable/8-unanswerable benchmark

You should understand grounding versus factual correctness, structured outputs, citation verification, context-window budgeting, temperature, retries, timeouts, and why retrieval confidence alone is not enough for abstention.

Likely interview questions:

- How do you verify that a citation actually supports a claim?
- How do you distinguish “retrieval failed” from “the corpus has no answer”?
- How do you prevent a paper from injecting instructions into the system prompt?
- Why might a stronger LLM fail to improve the final score?

## Phase 3 — Literature review and research ideation

Build:

- filters for year, method, hardware cost, accuracy, bit-stream source, and application
- comparison tables whose cells retain source citations
- a two-pass workflow: evidence extraction per paper, then cross-paper synthesis
- research-gap output separating observed evidence, inferred limitation, hypothesis, proposed experiment, and uncertainty

Acceptance criteria:

- comparison claims are traceable to individual papers rather than one mixed prompt
- conflicting numbers remain visible instead of being silently merged
- ideas include falsifiable experiments and do not claim novelty without evidence
- create a small qualitative rubric for traceability, novelty, feasibility, and uncertainty

You should understand map-reduce style synthesis, schema normalization, conflicting evidence, provenance, and why “prompting new ideas” needs a different evaluation from factual QA.

Likely interview questions:

- Why is one large prompt over 50 papers unreliable for literature review?
- How do you compare results measured under different technology nodes or datasets?
- How can the system propose ideas without pretending they are novel?
- When would a knowledge graph help more than vector retrieval?

## Phase 4 — Evaluation and observability

Build:

- atomic scoring for compound questions
- retrieval, answer, citation, and abstention metrics
- regression reports by question category
- traces for query rewrite, candidates, reranking, context, model output, latency, and token cost
- a compact error taxonomy and a repeatable experiment command

Acceptance criteria:

- one command reproduces the reported metrics
- every metric points to its dataset and configuration
- at least ten failures have documented causes and proposed fixes
- prompt/model/index versions appear in evaluation output

Likely interview questions:

- Why is LLM-as-judge not sufficient by itself?
- How do you make a RAG evaluation reproducible?
- Which metric would you optimize first and why?
- How do you detect regressions after changing chunking or prompts?

## Phase 5 — Product and deployment polish

Build:

- Streamlit research workspace with answer, evidence, paper comparison, and evaluation views
- Docker image and documented configuration
- request IDs, health/readiness endpoints, graceful model/API failures, and basic caching
- architecture diagram, demo script, screenshots, and final benchmark table

Acceptance criteria:

- a new machine can follow the README and reproduce the service
- the demo covers a normal answer, cross-paper comparison, and correct refusal
- failures return actionable messages without exposing secrets

Likely interview questions:

- How would this design change for one million documents or multiple users?
- What would you cache, and how would you invalidate it?
- How would you control latency and API cost?
- What production risks remain in the demo?

## Progression rule

Progress is based on understanding and acceptance criteria, not calendar days. Stay on the current phase until you can run its workflow, explain the key code path, interpret its metrics, and diagnose representative failures. Then move to the next phase. Resume writing can begin after Phase 2 because the project will already have a real corpus, measurable retrieval baseline, grounded generation, citations, and abstention. Phases 3–5 turn it into the stronger research-Agent project.

