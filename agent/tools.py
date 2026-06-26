import json
from pathlib import Path

import pymupdf
import requests
from errors import ErrorCode, ToolError, ToolException
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
    except ValueError as e:
        # Tavily's sync client raises ValueError (not HTTPError) on HTTP errors,
        # embedding the status code in the message, e.g. "Error 429: ...".
        if "429" in str(e):
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

class PDFPage(BaseModel):
    page_number: int
    text: str
    word_count: int

class Chunk(BaseModel):
    text: str
    page_number: int
    chunk_index: int

class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float

class PDFParser:
    def _fetch_from_url(self, source: str) -> bytes:
        """Fetch a PDF from a URL"""
        try:
            resp = requests.get(source, timeout=(10, 30))
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.Timeout as e:
            raise ToolException(ToolError(
                code=ErrorCode.TIMEOUT,
                message="Search timed out",
            )) from e
        except requests.exceptions.ConnectionError as e:
            raise ToolException(ToolError(
                code=ErrorCode.CONNECTION_ERROR,
                message="Failed to connect. Try again",
            )) from e
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ToolException(ToolError(
                    code=ErrorCode.FETCH_FAILED,
                    message="The current file can't be fetched. Please try another file.",
                )) from e
            raise ToolException(ToolError(
                code=ErrorCode.UNKNOWN,
                message=str(e),
            )) from e

    def _fetch_from_local(self, source: str) -> bytes:
        """Fetch a PDF from a file provided"""
        try:
            return Path(source).read_bytes()
        except FileNotFoundError as e:
            raise ToolException(ToolError(
                code=ErrorCode.FILE_NOT_FOUND,
                message=f"File not found at {source}. Please try another file.",
            )) from e
        except IsADirectoryError as e:
            raise ToolException(ToolError(
                code=ErrorCode.FILE_NOT_FOUND,
                message=f"{source} is a directory. Please try again with a file.",
            )) from e
        except OSError as e:
            raise ToolException(ToolError(
                code=ErrorCode.FETCH_FAILED,
                message=str(e),
            )) from e

    def extract_text(self, file_bytes: bytes) -> list[PDFPage]:
        """extract text from bytes and append each page to a List"""
        pages = []
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text()
                pages.append(
                    PDFPage(page_number=i, text=text, word_count=len(text.split()))
                )
        return pages

    def chunk(self, pages: list[PDFPage]) -> list[Chunk]:
        pass

    def score_chunks(self, query: str, chunks: list[Chunk]) -> list[ScoredChunk]:
        pass

    def run(self, source: str, query: str) -> list[ScoredChunk] | str:
        try:
            if source.startswith(("http://", "https://")):
                file_bytes = self._fetch_from_url(source)
            else:
                file_bytes = self._fetch_from_local(source)
            pages = self.extract_text(file_bytes)
            chunks = self.chunk(pages)
            return self.score_chunks(query, chunks)
        except ToolException as e:
            return e.error.model_dump_json()