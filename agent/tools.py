import ast
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pymupdf
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel

from agent.errors import ErrorCode, ToolError, ToolException
from agent.llm import get_embeddings_client

MAX_QUERY_LENGTH = 500

# Truncation limit for result snippets so a single search can't flood the
# context window.
MAX_SNIPPET_LENGTH = 500

# Retry schedule for rate-limited requests: sleep 1s, then 2s, then give up.
RATE_LIMIT_MAX_ATTEMPTS = 3

# Word-based proxy for ~500-token chunks with ~50-token overlap
# (English averages ~1.3 tokens/word).
CHUNK_SIZE_WORDS = 375
CHUNK_OVERLAP_WORDS = 37

# Number of top-ranked chunks score_chunks returns by default. Caps how much
# context flows downstream into the LLM's window.
TOP_K_DEFAULT = 8

# Hard cap on pages extracted from a single PDF so a huge document can't hang
# the tool or blow up embedding cost.
MAX_PDF_PAGES = 30


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, clamped to [0.0, 1.0].

    Negative similarity means unrelated, so we floor it at 0 to keep the score
    interpretable as 0 (unrelated) .. 1 (identical)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))

_client = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float


@tool
def web_search(query: str) -> str:
    """Search the web for information on a given topic."""
    if len(query) > MAX_QUERY_LENGTH:
        return ToolError(
            code=ErrorCode.QUERY_TOO_LONG,
            message=f"Query exceeds {MAX_QUERY_LENGTH} chars. Please shorten it.",
            tool_name="web_search",
        ).model_dump_json()

    for attempt in range(RATE_LIMIT_MAX_ATTEMPTS):
        try:
            results = _client.invoke(query)
            break
        except requests.exceptions.Timeout:
            return ToolError(
                code=ErrorCode.TIMEOUT,
                message="Search timed out. Try a simpler query.",
                tool_name="web_search",
            ).model_dump_json()
        except ValueError as e:
            # Tavily's sync client raises ValueError (not HTTPError) on HTTP errors,
            # embedding the status code in the message, e.g. "Error 429: ...".
            if "429" in str(e):
                if attempt < RATE_LIMIT_MAX_ATTEMPTS - 1:
                    time.sleep(2**attempt)
                    continue
                return ToolError(
                    code=ErrorCode.RATE_LIMIT,
                    message="Tavily rate limit reached. Retry after a short delay.",
                    tool_name="web_search",
                ).model_dump_json()
            return ToolError(
                code=ErrorCode.UNKNOWN,
                message=str(e),
                tool_name="web_search",
            ).model_dump_json()
        except Exception as e:
            return ToolError(
                code=ErrorCode.UNKNOWN,
                message=str(e),
                tool_name="web_search",
            ).model_dump_json()

    # Tavily returns a dict with "answer" and "results" keys; tolerate a bare
    # list of result dicts as well.
    raw_results = results.get("results") if isinstance(results, dict) else results
    if not raw_results:
        return ToolError(
            code=ErrorCode.EMPTY_RESULTS,
            message="No results found. Try rephrasing the query.",
            tool_name="web_search",
        ).model_dump_json()

    structured = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", "")[:MAX_SNIPPET_LENGTH],
            score=r.get("score", 0.0),
        )
        for r in raw_results
    ]
    answer = results.get("answer") if isinstance(results, dict) else None
    return json.dumps({
        "answer": answer,
        "results": [r.model_dump() for r in structured],
    })

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
    def __init__(self, embeddings: Embeddings | None = None):
        # Injectable so tests can pass a deterministic mock; defaults to the
        # backend selected by EMBEDDING_BACKEND.
        self._embeddings = embeddings or get_embeddings_client()

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
                tool_name="pdf_parser",
            )) from e
        except requests.exceptions.ConnectionError as e:
            raise ToolException(ToolError(
                code=ErrorCode.CONNECTION_ERROR,
                message="Failed to connect. Try again",
                tool_name="pdf_parser",
            )) from e
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ToolException(ToolError(
                    code=ErrorCode.FETCH_FAILED,
                    message="The current file can't be fetched. Please try another file.",
                    tool_name="pdf_parser",
                )) from e
            raise ToolException(ToolError(
                code=ErrorCode.UNKNOWN,
                message=str(e),
                tool_name="pdf_parser",
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
            if doc.needs_pass:
                raise ToolException(ToolError(
                    code=ErrorCode.PASSWORD_PROTECTED,
                    message="This PDF is password protected and can't be parsed."
                ))
            for i, page in enumerate(doc, start=1):
                if i > MAX_PDF_PAGES:
                    break
                text = page.get_text()
                pages.append(
                    PDFPage(page_number=i, text=text, word_count=len(text.split()))
                )
        total_words = sum(page.word_count for page in pages)
        if total_words < 200:
            raise ToolException(ToolError(
                code=ErrorCode.SCANNED_PDF_ERROR,
                message="Scanned PDF documents can't be parsed"
            ))
        return pages

    def chunk(self, pages: list[PDFPage]) -> list[Chunk]:
        """Split each page into ~500-token chunks (~50-token overlap) using a
        word-count proxy. chunk_index is sequential across the whole document."""
        step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
        chunks: list[Chunk] = []
        chunk_index = 0
        for page in pages:
            words = page.text.split()
            start = 0
            while start < len(words):
                window = words[start : start + CHUNK_SIZE_WORDS]
                chunks.append(Chunk(
                    text=" ".join(window),
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
                if start + CHUNK_SIZE_WORDS >= len(words):
                    break
                start += step
        return chunks

    def score_chunks(
        self, query: str, chunks: list[Chunk], top_k: int = TOP_K_DEFAULT
    ) -> list[ScoredChunk]:
        """Rank chunks by semantic relevance to the query via embedding cosine
        similarity. Returns the top_k chunks sorted by descending score."""
        if not chunks:
            return []
        query_vec = self._embeddings.embed_query(query)
        # Embed all chunk texts in one batched call rather than per-chunk.
        chunk_vecs = self._embeddings.embed_documents([c.text for c in chunks])
        scored = [
            ScoredChunk(chunk=chunk, score=_cosine(query_vec, vec))
            for chunk, vec in zip(chunks, chunk_vecs)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

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


# Created lazily so importing this module never constructs an embeddings
# client (the real backend needs env vars that only exist at runtime).
_default_pdf_parser: PDFParser | None = None


def _get_pdf_parser() -> PDFParser:
    global _default_pdf_parser
    if _default_pdf_parser is None:
        _default_pdf_parser = PDFParser()
    return _default_pdf_parser


@tool
def pdf_parser(source: str, query: str) -> str:
    """Extract the passages of a PDF most relevant to a query.

    source: a URL (http/https) or a local file path pointing at a PDF.
    query: what to look for; the most relevant passages are returned first.
    """
    result = _get_pdf_parser().run(source, query)
    if isinstance(result, str):
        return result
    return json.dumps([scored.model_dump() for scored in result])


# Hard cap on snippet size: agent-generated code should be short; anything
# bigger is almost certainly a prompt gone wrong.
MAX_CODE_LENGTH = 2000

EXEC_TIMEOUT_SECONDS = 10

# Truncation limit for captured stdout/stderr so a print-loop can't flood
# the context window.
MAX_OUTPUT_CHARS = 2000

EXEC_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

# Modules that allow shell access, networking, or filesystem/process
# manipulation. Blocking the root module also blocks its submodules
# (e.g. blocking os blocks os.system).
BLOCKED_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "importlib",
})


def _find_blocked_import(code: str) -> str | None:
    """Return the first blocked module imported by the code, or None.

    Raises SyntaxError if the code doesn't parse — the caller maps that to
    a ToolError before ever spawning a subprocess.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    return root
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BLOCKED_MODULES:
                return root
        elif isinstance(node, ast.Name) and node.id == "__import__":
            # __import__("subprocess") would bypass the static import check.
            return "__import__"
    return None


def _apply_resource_limits() -> None:
    """Cap CPU time and memory in the child process (preexec_fn)."""
    import resource

    # One second looser than the wall-clock timeout so subprocess.run's
    # TimeoutExpired fires first; the kernel CPU limit is only a backstop.
    cpu_limit = EXEC_TIMEOUT_SECONDS + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES)
        )
    except ValueError:
        # macOS doesn't reliably support RLIMIT_AS; CPU limit still applies.
        pass


@tool
def run_python(code: str) -> str:
    """Execute a short Python snippet and return its stdout and stderr.

    The code runs in an isolated subprocess with a 10-second timeout and no
    access to os, sys, subprocess, socket, or other system modules. Use
    print() to emit results.
    """
    if len(code) > MAX_CODE_LENGTH:
        return ToolError(
            code=ErrorCode.CODE_TOO_LONG,
            message=f"Code exceeds {MAX_CODE_LENGTH} chars. Send a shorter snippet.",
            tool_name="run_python",
        ).model_dump_json()

    try:
        blocked = _find_blocked_import(code)
    except SyntaxError as e:
        return ToolError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Code has a syntax error: {e}",
            tool_name="run_python",
        ).model_dump_json()
    if blocked is not None:
        return ToolError(
            code=ErrorCode.BLOCKED_IMPORT,
            message=f"Import of '{blocked}' is not allowed in the sandbox.",
            tool_name="run_python",
        ).model_dump_json()

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            preexec_fn=_apply_resource_limits,
        )
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return ToolError(
            code=ErrorCode.TIMEOUT,
            message=(
                f"Execution exceeded {EXEC_TIMEOUT_SECONDS}s and was killed. "
                f"Partial stdout: {partial[:MAX_OUTPUT_CHARS]}"
            ),
            tool_name="run_python",
        ).model_dump_json()

    stdout = completed.stdout[:MAX_OUTPUT_CHARS]
    stderr = completed.stderr[:MAX_OUTPUT_CHARS]
    if completed.returncode == -signal.SIGXCPU:
        # Killed by the RLIMIT_CPU backstop rather than the wall-clock timeout.
        return ToolError(
            code=ErrorCode.TIMEOUT,
            message=(
                f"Execution exceeded the CPU limit and was killed. "
                f"Partial stdout: {stdout}"
            ),
            tool_name="run_python",
        ).model_dump_json()
    if completed.returncode != 0:
        return ToolError(
            code=ErrorCode.EXECUTION_FAILED,
            message=f"Code exited with status {completed.returncode}. stderr: {stderr}",
            tool_name="run_python",
        ).model_dump_json()
    return json.dumps({"stdout": stdout, "stderr": stderr, "exit_code": 0})


HF_API_BASE = "https://huggingface.co/api"

MAX_HUB_RESULTS = 5


class HubResult(BaseModel):
    name: str
    description: str
    downloads: int
    url: str


@tool
def huggingface_search(
    query: str, search_type: str = "models", task: str | None = None
) -> str:
    """Search the HuggingFace Hub for models or datasets.

    search_type: "models" or "datasets".
    task: optional pipeline tag (e.g. "text-classification", "summarization")
    to filter models by task type.
    """
    if search_type not in ("models", "datasets"):
        return ToolError(
            code=ErrorCode.INVALID_INPUT,
            message='search_type must be "models" or "datasets".',
            tool_name="huggingface_search",
        ).model_dump_json()

    params: dict = {
        "search": query,
        "limit": MAX_HUB_RESULTS,
        "sort": "downloads",
        "direction": -1,
    }
    if task and search_type == "models":
        params["pipeline_tag"] = task
    # Token is optional: anonymous requests work with lower rate limits.
    headers = {}
    if os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

    try:
        resp = requests.get(
            f"{HF_API_BASE}/{search_type}",
            params=params,
            headers=headers,
            timeout=(10, 30),
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return ToolError(
            code=ErrorCode.TIMEOUT,
            message="HuggingFace Hub request timed out. Try again.",
            tool_name="huggingface_search",
        ).model_dump_json()
    except requests.exceptions.ConnectionError:
        return ToolError(
            code=ErrorCode.CONNECTION_ERROR,
            message="Failed to connect to HuggingFace Hub. Try again.",
            tool_name="huggingface_search",
        ).model_dump_json()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 429:
            return ToolError(
                code=ErrorCode.RATE_LIMIT,
                message="HuggingFace Hub rate limit reached. Retry after a delay.",
                tool_name="huggingface_search",
            ).model_dump_json()
        if status in (401, 403):
            return ToolError(
                code=ErrorCode.AUTH_ERROR,
                message="HuggingFace Hub rejected the request. Check HF_TOKEN.",
                tool_name="huggingface_search",
            ).model_dump_json()
        return ToolError(
            code=ErrorCode.FETCH_FAILED,
            message=f"HuggingFace Hub returned an error: {e}",
            tool_name="huggingface_search",
        ).model_dump_json()

    items = resp.json()
    if not items:
        return ToolError(
            code=ErrorCode.EMPTY_RESULTS,
            message="No results found on the Hub. Try a broader query.",
            tool_name="huggingface_search",
        ).model_dump_json()

    url_prefix = "datasets/" if search_type == "datasets" else ""
    results = [
        HubResult(
            name=item.get("id", ""),
            description=(item.get("description") or "")[:MAX_SNIPPET_LENGTH],
            downloads=item.get("downloads") or 0,
            url=f"https://huggingface.co/{url_prefix}{item.get('id', '')}",
        )
        for item in items[:MAX_HUB_RESULTS]
    ]
    return json.dumps({"results": [r.model_dump() for r in results]})