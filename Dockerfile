FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml Makefile README.md ./
COPY src ./src
COPY scripts ./scripts
COPY ui ./ui

RUN python -m pip install --upgrade pip \
    && python -m pip install '.[hybrid,ui]'

EXPOSE 8000 8501

CMD ["uvicorn", "sc_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
