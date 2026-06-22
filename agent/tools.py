import json
from enum import Enum
from errors import ErrorCode, ToolError

import requests.exceptions
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel

MAX_QUERY_LENGTH = 500

_client = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)


@tool
def web_search(query: str) -> str:
    """Search the web for information on a given topic."""
    if len(query) > MAX_QUERY_LENGTH:
        return ToolError(
            code=ErrorCode.QUERY_TOO_LONG,
            message=f"Query exceeds {MAX_QUERY_LENGTH} chars. Please shorten it.",
        ).model_dump_json()

    try:
        results = _client.invoke(query)
    except requests.exceptions.Timeout:
        return ToolError(
            code=ErrorCode.TIMEOUT,
            message="Search timed out. Try a simpler query.",
        ).model_dump_json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            return ToolError(
                code=ErrorCode.RATE_LIMIT,
                message="Tavily rate limit reached. Retry after a short delay.",
            ).model_dump_json()
        return ToolError(
            code=ErrorCode.UNKNOWN,
            message=str(e),
        ).model_dump_json()
    except Exception as e:
        return ToolError(
            code=ErrorCode.UNKNOWN,
            message=str(e),
        ).model_dump_json()

    if not results:
        return ToolError(
            code=ErrorCode.EMPTY_RESULTS,
            message="No results found. Try rephrasing the query.",
        ).model_dump_json()

    return json.dumps(results)


