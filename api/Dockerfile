FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn python-dotenv

COPY api/ .

CMD uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8000}
