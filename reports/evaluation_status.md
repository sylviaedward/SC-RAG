# Evaluation status

This repository tracks the evaluation workflow and its current reproducibility status. Generated model outputs, indexes, and raw reports stay ignored because they depend on local corpus files and credentials.

## Checks verified in the repository

- Benchmark contract: 58 questions total — 50 answerable and 8 unanswerable.
- Unit and contract suite: 47 tests collected; 46 passed and 1 skipped because `artifacts/chunks.jsonl` is not present in a clean checkout.
- The skipped check is the generated chunk identity round trip; it is enabled after corpus ingestion.

## Full evaluation

The phase reports are produced with:

```bash
make evaluate-phase4
python scripts/run_reranker_ablation.py
```

Those commands require the generated corpus and dense index under `artifacts/`, plus an OpenAI-compatible model configuration for answer evaluation. A clean checkout does not contain those inputs, so this repository does not claim Top-5 recall, answer accuracy, abstention accuracy, or latency values until a run produces the corresponding JSON files.

Expected generated files are `artifacts/phase4_eval.json` and `reports/reranker_ablation.json`. Do not enter placeholder values when those files are absent.
