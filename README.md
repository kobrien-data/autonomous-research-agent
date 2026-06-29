# Autonomous Research Agent

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C)
![LangChain](https://img.shields.io/badge/LangChain-agent_framework-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral-LLM-FF7000?logo=mistralai&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-code_LLM-4D6BFE?logo=deepseek&logoColor=white)
![RunPod](https://img.shields.io/badge/RunPod-GPU_hosting-673AB7?logo=runpod&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-inference-FFB000)
![Docker](https://img.shields.io/badge/Docker-deploy-2496ED?logo=docker&logoColor=white)

## Overview

**Autonomous Research Agent** is a multi-agent research system that takes a
natural-language query, plans how to answer it, and coordinates a team of specialist
agents to gather and synthesise evidence, with no step-by-step hand-holding required.

At its core is a **supervisor agent**, built on **LangGraph**, that plans, routes, and
synthesises the final answer. It delegates work to four specialist agents, each backed by
its own tool:

- **Search agent**: query planning and result ranking, powered by
  [Tavily](https://tavily.com/) web search.
- **Document agent**: PDF parsing and chunk retrieval via
  [PyMuPDF](https://pymupdf.readthedocs.io/).
- **Code agent**: writes and executes Python in a subprocess sandbox.
- **Database agent**: logs message history, token usage, and costs to SQLite through an
  **MCP server** (JSON-RPC over stdio).

All agents share a **LangGraph message channel**: they read and write observations to a
common state, the supervisor sees the full picture, and the database agent persists that
history to SQLite via MCP.

Models are self-hosted on **RunPod** with **vLLM**, mixing several open-weight models to
balance capability against cost: a Mistral-7B-Instruct instance for the supervisor and
synthesis, a faster lower-temperature Mistral-7B shared by the search, document, and
database agents, and DeepSeek-Coder-7B for the code agent. Weights are pulled from the
HuggingFace Hub.

The system is served behind a **FastAPI** backend (`/query`, `/history`, SSE streaming)
and a **Streamlit** UI featuring a chat interface and an agent trace panel. The full stack
ships as **Docker** containers and can be brought up with a single `docker compose up`.

## Architecture

> _TODO_

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose

### Run the full stack

```bash
docker compose up --build
```

| Service | Default URL |
|---------|-------------|
| API     | http://localhost:8000 |
| UI      | http://localhost:8501 |

Configuration is read from `.env`. Copy `.env` and adjust ports or LLM settings before starting:

```bash
# Change ports without touching source code
API_PORT=9000 UI_PORT=9501 docker compose up --build
```

The UI waits for the API health check to pass before starting.

### Run Tests
```bash
uv run pytest tests/tools-test.py -v
```

## Usage

> _TODO_

## Roadmap

> _TODO_

## License

> _TODO_
