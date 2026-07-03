import json
from unittest.mock import MagicMock

import pymupdf
import pytest
import requests.exceptions

import agent.tools as tools
from agent.llm import MockEmbeddings
from agent.tools import (
    ErrorCode,
    PDFParser,
    huggingface_search,
    pdf_parser,
    run_python,
    web_search,
)

# ---------------------------------------------------------------------------
# web_search (W2-01)
# ---------------------------------------------------------------------------

#https://docs.python.org/3/library/unittest.mock.html
def fake_tavily(monkeypatch, *, result=None, exc=None):
    """Swap the module-global Tavily client for a MagicMock.

    web_search reads the module global _client at call time, so replacing it
    here routes the call through our mock. monkeypatch restores it after the test.
    """
    client = MagicMock()
    if exc is not None:
        client.invoke.side_effect = exc
    else:
        client.invoke.return_value = result
    monkeypatch.setattr(tools, "_client", client)
    return client


def run_search(query: str = "what is langgraph") -> dict:
    """Invoke the tool and parse its JSON string response."""
    return json.loads(web_search.invoke({"query": query}))


def test_search_success_path(monkeypatch):
    fake_tavily(monkeypatch, result={
        "answer": "LangGraph is a library for building stateful agents.",
        "results": [
            {
                "title": "LangGraph",
                "url": "https://example.com",
                "content": "x" * 600,
                "score": 0.9,
            }
        ],
    })

    response = run_search()

    assert response["answer"].startswith("LangGraph is")
    assert response["results"][0]["title"] == "LangGraph"
    assert response["results"][0]["url"] == "https://example.com"
    assert response["results"][0]["score"] == 0.9
    # Snippets are truncated to protect the context window.
    assert len(response["results"][0]["snippet"]) == 500


def test_search_empty_results(monkeypatch):
    fake_tavily(monkeypatch, result=[])

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.EMPTY_RESULTS.value


def test_search_rate_limit_retries_then_errors(monkeypatch):
    # Tavily's sync client surfaces HTTP errors as ValueError with the status
    # code embedded in the message.
    client = fake_tavily(monkeypatch, exc=ValueError("Error 429: rate limited"))
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.RATE_LIMIT.value
    assert response["retryable"] is True
    # Exponential backoff: 1s then 2s before giving up on the third attempt.
    assert client.invoke.call_count == 3
    assert sleeps == [1, 2]


def test_search_timeout(monkeypatch):
    fake_tavily(monkeypatch, exc=requests.exceptions.Timeout())

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.TIMEOUT.value
    assert response["retryable"] is True


def test_search_query_too_long():
    response = run_search("x" * 501)

    assert response["error"] is True
    assert response["code"] == ErrorCode.QUERY_TOO_LONG.value
    assert response["retryable"] is False


def test_search_unknown_error(monkeypatch):
    fake_tavily(monkeypatch, exc=RuntimeError("boom"))

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.UNKNOWN.value


# ---------------------------------------------------------------------------
# pdf_parser (W2-02)
# ---------------------------------------------------------------------------

def make_pdf(pages: list[str]) -> bytes:
    """Build an in-memory PDF with one page per string.

    Tiny font so several hundred words fit inside the page box — words
    rendered outside it are dropped by the text extractor.
    """
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((36, 36), text, fontsize=5)
    return doc.tobytes()


def make_page_text(n_words: int, word: str = "alpha") -> str:
    """n_words distinct words, wrapped so they fit on a PDF page's text layer."""
    words = [f"{word}{i}" for i in range(n_words)]
    lines = [" ".join(words[i : i + 8]) for i in range(0, len(words), 8)]
    return "\n".join(lines)


@pytest.fixture
def parser() -> PDFParser:
    return PDFParser(embeddings=MockEmbeddings())


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf([make_page_text(300)])


def fake_http_get(monkeypatch, *, content=None, status=200, exc=None):
    """Replace requests.get as seen by agent.tools with a canned response."""
    calls = {}

    def _get(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        if exc is not None:
            raise exc
        resp = MagicMock()
        resp.content = content
        resp.status_code = status
        if status >= 400:
            http_response = requests.models.Response()
            http_response.status_code = status
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                response=http_response
            )
        else:
            resp.raise_for_status.return_value = None
        resp.json.return_value = content
        return resp

    monkeypatch.setattr(tools.requests, "get", _get)
    return calls


def test_pdf_run_local_success(tmp_path, parser, pdf_bytes):
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes)

    scored = parser.run(str(path), "alpha5")

    assert isinstance(scored, list)
    assert len(scored) >= 1
    assert scored[0].chunk.page_number == 1
    assert "alpha5" in scored[0].chunk.text


def test_pdf_run_url_dispatch(monkeypatch, parser, pdf_bytes):
    calls = fake_http_get(monkeypatch, content=pdf_bytes)

    scored = parser.run("https://example.com/doc.pdf", "alpha5")

    assert calls["url"] == "https://example.com/doc.pdf"
    assert isinstance(scored, list)


def test_pdf_url_404(monkeypatch, parser):
    fake_http_get(monkeypatch, status=404)

    response = json.loads(parser.run("https://example.com/gone.pdf", "q"))

    assert response["error"] is True
    assert response["code"] == ErrorCode.FETCH_FAILED.value


def test_pdf_url_timeout(monkeypatch, parser):
    fake_http_get(monkeypatch, exc=requests.exceptions.Timeout())

    response = json.loads(parser.run("https://example.com/slow.pdf", "q"))

    assert response["code"] == ErrorCode.TIMEOUT.value
    assert response["retryable"] is True


def test_pdf_file_not_found(parser):
    response = json.loads(parser.run("/nope/missing.pdf", "q"))

    assert response["code"] == ErrorCode.FILE_NOT_FOUND.value
    assert response["retryable"] is False


def test_pdf_scanned_detection(tmp_path, parser):
    # Fewer than 200 words total means no usable text layer (scanned doc).
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf(["just a few words"]))

    response = json.loads(parser.run(str(path), "q"))

    assert response["code"] == ErrorCode.SCANNED_PDF_ERROR.value


def test_pdf_page_cap(parser):
    pages = parser.extract_text(make_pdf([make_page_text(50)] * 35))

    assert len(pages) == tools.MAX_PDF_PAGES


def test_pdf_chunk_size_and_overlap(parser):
    pages = parser.extract_text(make_pdf([make_page_text(400), make_page_text(800)]))

    chunks = parser.chunk(pages)

    first_page_chunks = [c for c in chunks if c.page_number == 1]
    second_page_chunks = [c for c in chunks if c.page_number == 2]
    assert len(first_page_chunks) == 2
    assert len(second_page_chunks) == 3
    # chunk_index is sequential across the whole document.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Consecutive chunks on a page share the overlap window.
    a, b = second_page_chunks[0], second_page_chunks[1]
    assert len(a.text.split()) == tools.CHUNK_SIZE_WORDS
    overlap = tools.CHUNK_OVERLAP_WORDS
    assert a.text.split()[-overlap:] == b.text.split()[:overlap]


def test_pdf_score_chunks_ranks_relevant_first(parser):
    pages = parser.extract_text(
        make_pdf([make_page_text(150, "noise") + "\n" + make_page_text(150, "signal")])
    )
    chunks = parser.chunk(pages)

    scored = parser.score_chunks("signal1 signal2 signal3", chunks, top_k=3)

    assert len(scored) <= 3
    assert scored[0].score >= scored[-1].score
    assert "signal1" in scored[0].chunk.text


def test_pdf_tool_wrapper_returns_json(tmp_path, monkeypatch, pdf_bytes):
    monkeypatch.setattr(tools, "_default_pdf_parser", PDFParser(MockEmbeddings()))
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes)

    response = json.loads(pdf_parser.invoke({"source": str(path), "query": "alpha5"}))

    assert isinstance(response, list)
    assert {"chunk", "score"} <= response[0].keys()


# ---------------------------------------------------------------------------
# run_python (W2-03)
# ---------------------------------------------------------------------------

def run_code(code: str) -> dict:
    return json.loads(run_python.invoke({"code": code}))


def test_code_success():
    response = run_code("print('hello')")

    assert response["exit_code"] == 0
    assert response["stdout"] == "hello\n"
    assert response["stderr"] == ""


def test_code_timeout(monkeypatch):
    monkeypatch.setattr(tools, "EXEC_TIMEOUT_SECONDS", 1)

    response = run_code("while True: pass")

    assert response["error"] is True
    assert response["code"] == ErrorCode.TIMEOUT.value
    assert response["retryable"] is True


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess",
        "import socket",
        "from os import system",
        "import os.path",
        "__import__('subprocess')",
    ],
)
def test_code_blocked_import(code):
    response = run_code(code)

    assert response["error"] is True
    assert response["code"] == ErrorCode.BLOCKED_IMPORT.value
    assert response["retryable"] is False


def test_code_syntax_error():
    response = run_code("def f(:")

    assert response["error"] is True
    assert response["code"] == ErrorCode.INVALID_INPUT.value


def test_code_too_long():
    response = run_code("x = 1\n" * 400)

    assert response["error"] is True
    assert response["code"] == ErrorCode.CODE_TOO_LONG.value


def test_code_runtime_error():
    response = run_code("raise ValueError('boom')")

    assert response["error"] is True
    assert response["code"] == ErrorCode.EXECUTION_FAILED.value
    assert "boom" in response["message"]


def test_code_allowed_import_works():
    response = run_code("import math\nprint(math.pi)")

    assert response["exit_code"] == 0
    assert response["stdout"].startswith("3.14")


# ---------------------------------------------------------------------------
# huggingface_search (W2-04)
# ---------------------------------------------------------------------------

def run_hub(**kwargs) -> dict:
    return json.loads(huggingface_search.invoke(kwargs))


def test_hub_model_search(monkeypatch):
    calls = fake_http_get(monkeypatch, content=[
        {"id": "org/model-a", "downloads": 100, "description": "A model"},
        {"id": "org/model-b", "downloads": 50},
    ])

    response = run_hub(query="sentiment", search_type="models")

    assert calls["url"].endswith("/api/models")
    names = [r["name"] for r in response["results"]]
    assert names == ["org/model-a", "org/model-b"]
    assert response["results"][0]["url"] == "https://huggingface.co/org/model-a"
    assert response["results"][1]["description"] == ""


def test_hub_dataset_search(monkeypatch):
    calls = fake_http_get(monkeypatch, content=[{"id": "squad", "downloads": 5}])

    response = run_hub(query="qa", search_type="datasets")

    assert calls["url"].endswith("/api/datasets")
    assert response["results"][0]["url"] == "https://huggingface.co/datasets/squad"


def test_hub_pipeline_tag_filter(monkeypatch):
    calls = fake_http_get(monkeypatch, content=[{"id": "m", "downloads": 1}])

    run_hub(query="ner", search_type="models", task="token-classification")

    assert calls["kwargs"]["params"]["pipeline_tag"] == "token-classification"


def test_hub_top_five_cap(monkeypatch):
    fake_http_get(
        monkeypatch,
        content=[{"id": f"m{i}", "downloads": i} for i in range(10)],
    )

    response = run_hub(query="x")

    assert len(response["results"]) == 5


def test_hub_api_error(monkeypatch):
    fake_http_get(monkeypatch, status=500)

    response = run_hub(query="x")

    assert response["error"] is True
    assert response["code"] == ErrorCode.FETCH_FAILED.value


def test_hub_rate_limit(monkeypatch):
    fake_http_get(monkeypatch, status=429)

    response = run_hub(query="x")

    assert response["code"] == ErrorCode.RATE_LIMIT.value
    assert response["retryable"] is True


def test_hub_auth_error_not_retryable(monkeypatch):
    fake_http_get(monkeypatch, status=401)

    response = run_hub(query="x")

    assert response["code"] == ErrorCode.AUTH_ERROR.value
    assert response["retryable"] is False


def test_hub_empty_results(monkeypatch):
    fake_http_get(monkeypatch, content=[])

    response = run_hub(query="zzz")

    assert response["code"] == ErrorCode.EMPTY_RESULTS.value


def test_hub_invalid_search_type():
    response = run_hub(query="x", search_type="spaces")

    assert response["code"] == ErrorCode.INVALID_INPUT.value
    assert response["retryable"] is False


def test_hub_token_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secrettoken")
    calls = fake_http_get(monkeypatch, content=[{"id": "m", "downloads": 1}])

    run_hub(query="x")

    assert calls["kwargs"]["headers"]["Authorization"] == "Bearer hf_secrettoken"


def test_hub_works_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = fake_http_get(monkeypatch, content=[{"id": "m", "downloads": 1}])

    response = run_hub(query="x")

    assert "Authorization" not in calls["kwargs"]["headers"]
    assert response["results"][0]["name"] == "m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
