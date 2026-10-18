PYTHON := .venv/bin/python

.PHONY: system-deps setup import-corpus manifest ingest test search evaluate api

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

import-corpus:
	@test -n "$(ARCHIVES)" || (echo 'Usage: make import-corpus ARCHIVES="imports/SC_Liter*.zip"' && exit 2)
	$(PYTHON) scripts/import_corpus_archives.py $(ARCHIVES)

manifest:
	@command -v pdfinfo >/dev/null || (echo 'Missing pdfinfo. Run: make system-deps' && exit 2)
	@command -v pdftotext >/dev/null || (echo 'Missing pdftotext. Run: make system-deps' && exit 2)
	$(PYTHON) scripts/build_corpus_manifest.py

ingest:
	$(PYTHON) -m sc_rag.ingest

test:
	$(PYTHON) -m unittest discover -s tests -v

search:
	@test -n "$(QUERY)" || (echo 'Usage: make search QUERY="your question"' && exit 2)
	$(PYTHON) -m sc_rag.search "$(QUERY)" --limit 5

evaluate:
	$(PYTHON) scripts/evaluate_retrieval.py --k 10

api:
	$(PYTHON) -m uvicorn sc_rag.api:app --host 0.0.0.0 --port 8000

