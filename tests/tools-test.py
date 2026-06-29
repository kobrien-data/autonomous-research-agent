import json
from unittest.mock import MagicMock

import pytest
import requests.exceptions

import agent.errors as errors
import agent.llm as llm
import agent.tools as tools
from agent.tools import ErrorCode, web_search

#https://docs.python.org/3/library/unittest.mock.html
def fake_client(monkeypatch, *, result=None, exc=None):
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


def test_success_path(monkeypatch):
    results = {
        "answer": "LangGraph is a library for building stateful agents.",
        "results": [
            {"title": "LangGraph", "url": "https://example.com", "content": "..."}
        ],
    }
    fake_client(monkeypatch, result=results)

    response = run_search()

    assert response == results
    assert "error" not in response


def test_empty_results(monkeypatch):
    fake_client(monkeypatch, result=[])

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.EMPTY_RESULTS.value


def test_rate_limit(monkeypatch):
    http_response = requests.models.Response()
    http_response.status_code = 429
    fake_client(monkeypatch, exc=requests.exceptions.HTTPError(response=http_response))

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.RATE_LIMIT.value


def test_timeout(monkeypatch):
    fake_client(monkeypatch, exc=requests.exceptions.Timeout())

    response = run_search()

    assert response["error"] is True
    assert response["code"] == ErrorCode.TIMEOUT.value


def test_query_too_long():
    response = json.loads(web_search.invoke({"query": "x" * 501}))

    assert response["error"] is True
    assert response["code"] == ErrorCode.QUERY_TOO_LONG.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
