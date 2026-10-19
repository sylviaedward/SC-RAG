# Phase 1 initial retrieval results

Configuration:

- Corpus: 50 SC papers, 2,216 page-aware chunks
- Evaluation: 50 answerable draft questions, Top-10
- Dense model: `intfloat/multilingual-e5-small`
- Fusion: RRF with `candidate_k=40`, `rrf_k=60`
- Diversity: at most two chunks per paper
- Reranker smoke test: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `rerank_k=10`, one CPU thread

| Retriever | Hit-any@10 | Hit-all@10 | MRR | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| BM25 + bilingual expansion | 0.86 | 0.58 | 0.7569 | 14.95 ms | 21.96 ms |
| multilingual-E5 dense | 0.52 | 0.26 | 0.2689 | 115.96 ms | 187.22 ms |
| Hybrid RRF | **0.94** | **0.68** | 0.7548 | 97.70 ms | 150.00 ms |
| Hybrid + reranker smoke | 0.90 | 0.60 | 0.7309 | 4,253.62 ms | 4,497.07 ms |

## Interpretation

Dense retrieval is weak as a standalone retriever on this benchmark, but it contributes evidence that lexical retrieval misses. RRF improves Hit-any by 8 percentage points and Hit-all by 10 points over BM25 while leaving MRR nearly unchanged. This is the intended hybrid behavior: each route covers failures from the other route.

Several properties favor BM25 in this draft benchmark. Many questions explicitly name a paper alias, formulas contain distinctive terms, and bilingual expansion supplies important English SC terminology. The papers are English while questions are Chinese and often compound, which is difficult for a small general multilingual embedding model. These conditions must be reported rather than treating one aggregate score as a universal model ranking.

The `rerank_k=10` result validates the Cross-Encoder execution path but is not the formal reranker comparison. Diversity filtering occurs after reranking; if the ten candidates contain repeated papers, the final response can contain fewer than ten hits. The evaluator now reports `mean_returned_hits` and `full_k_rate` to expose this condition. The formal experiment uses `rerank_k=30` and compares failures, latency, and result coverage before deciding whether to retain, replace, or selectively invoke the reranker.

## Next checks

1. Run the formal reranker experiment with `rerank_k=30`.
2. Compare `mean_returned_hits` and `full_k_rate` before interpreting recall differences.
3. Inspect examples in which Dense rescues BM25 and in which Reranker harms RRF.
4. Test whether translating compound queries or using a domain-adapted reranker improves cross-language ranking.

