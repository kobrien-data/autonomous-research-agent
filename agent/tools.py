import json
from pathlib import Path

import pymupdf
import requests
from errors import ErrorCode, ToolError, ToolException
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel

MAX_QUERY_LENGTH = 500

# Word-based proxy for ~500-token chunks with ~50-token overlap
# (English averages ~1.3 tokens/word).
CHUNK_SIZE_WORDS = 375
CHUNK_OVERLAP_WORDS = 37

# Number of top-ranked chunks score_chunks returns by default. Caps how much
# context flows downstream into the LLM's window.
TOP_K_DEFAULT = 8


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
            if doc.needs_pass:
                raise ToolException(ToolError(
                    code=ErrorCode.PASSWORD_PROTECTED,
                    message="This PDF is password protected and can't be parsed."
                ))
            for i, page in enumerate(doc, start=1):
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