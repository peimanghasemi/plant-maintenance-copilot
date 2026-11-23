FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[llm]"
COPY docs ./docs
COPY scripts ./scripts
RUN python scripts/train.py
EXPOSE 8000
CMD ["uvicorn", "plantcopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
