PYTHON := .venv/bin/python
PORT ?= 8000
EMBEDDING_MODEL ?= intfloat/multilingual-e5-small
EMBEDDING_PROVIDER ?= local
RERANKER_MODEL ?= cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

.PHONY: system-deps setup setup-hybrid import-corpus manifest ingest dense-index test search dense-search hybrid-search-rrf hybrid-search evaluate evaluate-phase1 api

system-deps:
	@if command -v pdfinfo >/dev/null && command -v pdftotext >/dev/null; then \
		echo 'poppler-utils is already installed.'; \
	else \
		sudo apt-get update && sudo apt-get install -y poppler-utils; \
	fi

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

setup-hybrid:
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e '.[hybrid]'

import-corpus:
	@test -n "$(ARCHIVES)" || (echo 'Usage: make import-corpus ARCHIVES="imports/SC_Liter*.zip"' && exit 2)
	$(PYTHON) scripts/import_corpus_archives.py $(ARCHIVES)

manifest:
	@command -v pdfinfo >/dev/null || (echo 'Missing pdfinfo. Run: make system-deps' && exit 2)
	@command -v pdftotext >/dev/null || (echo 'Missing pdftotext. Run: make system-deps' && exit 2)
	$(PYTHON) scripts/build_corpus_manifest.py

ingest:
	$(PYTHON) -m sc_rag.ingest

dense-index:
	$(PYTHON) scripts/build_dense_index.py --provider "$(EMBEDDING_PROVIDER)" --model "$(EMBEDDING_MODEL)"

test:
	$(PYTHON) -m unittest discover -s tests -v

search:
	@test -n "$(QUERY)" || (echo 'Usage: make search QUERY="your question"' && exit 2)
	$(PYTHON) -m sc_rag.search "$(QUERY)" --limit 5

dense-search:
	@test -n "$(QUERY)" || (echo 'Usage: make dense-search QUERY="your question"' && exit 2)
	$(PYTHON) -m sc_rag.dense "$(QUERY)" --provider "$(EMBEDDING_PROVIDER)" --model "$(EMBEDDING_MODEL)" --limit 5

hybrid-search-rrf:
	@test -n "$(QUERY)" || (echo 'Usage: make hybrid-search-rrf QUERY="your question"' && exit 2)
	$(PYTHON) -m sc_rag.hybrid "$(QUERY)" --embedding-provider "$(EMBEDDING_PROVIDER)" --embedding-model "$(EMBEDDING_MODEL)" --no-reranker --limit 5

hybrid-search:
	@test -n "$(QUERY)" || (echo 'Usage: make hybrid-search QUERY="your question"' && exit 2)
	$(PYTHON) -m sc_rag.hybrid "$(QUERY)" --embedding-provider "$(EMBEDDING_PROVIDER)" --embedding-model "$(EMBEDDING_MODEL)" --reranker-model "$(RERANKER_MODEL)" --limit 5

evaluate:
	$(PYTHON) scripts/evaluate_retrieval.py --k 10

evaluate-phase1:
	$(PYTHON) scripts/evaluate_phase1.py --embedding-provider "$(EMBEDDING_PROVIDER)" --embedding-model "$(EMBEDDING_MODEL)" --reranker-model "$(RERANKER_MODEL)" --k 10

api:
	$(PYTHON) -m uvicorn sc_rag.api:app --host 0.0.0.0 --port $(PORT)

