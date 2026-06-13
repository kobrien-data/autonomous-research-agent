FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests python-dotenv

COPY ui/ .

CMD streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port ${UI_PORT:-8501} \
    --server.headless true
