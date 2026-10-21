# Chunking policy

SC-RAG extracts text page by page and stores `paper_id`, alias, PDF page, and
chunk identity with every chunk. Page-aware chunks preserve the citation unit
that a researcher can open in the source PDF and prevent a claim from being
grounded only by text copied from a neighboring page.

The ingestion code currently uses a 1,400-character window with a
180-character overlap inside each page. The window is large enough to keep a
definition, result, or table note together, while the overlap protects
sentences crossing a boundary. The
answering layer applies a second budget (`ContextBudget`) and splits selected
chunks into short trusted spans. This two-level policy keeps retrieval recall
and prompt size separate: ingestion controls searchable units, while answering
controls model context and citation granularity.

When changing chunk size or overlap, rerun ingestion and the same benchmark.
Do not compare metrics from indexes built with different manifests without
recording the index configuration in the report.
