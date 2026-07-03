import hashlib
import json
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# Repo-relative so the fixture resolves no matter the working directory.
MOCK_RESPONSES_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mock_responses.json"
)


def _require_env(name: str) -> str:
    """Read a required env var, failing loudly if it's missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set for this backend.")
    return value


class MockLLMClient(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        responses = json.loads(MOCK_RESPONSES_PATH.read_text())
        content = responses[0]["content"]
        response = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=response)])


def _get_vllm_client() -> ChatOpenAI:
    return ChatOpenAI(
        model=_require_env("VLLM_MODEL_NAME"),
        base_url=_require_env("VLLM_BASE_URL"),
        api_key=SecretStr(_require_env("VLLM_API_KEY")),
        temperature=float(os.getenv("VLLM_TEMPERATURE", "0.7")),
        max_completion_tokens=int(os.getenv("VLLM_MAX_TOKENS", "1024")),
    )



def get_llm_client() -> BaseChatModel:
    backend = os.getenv("LLM_BACKEND", "mock")
    if backend == "vllm":
        return _get_vllm_client()
    return MockLLMClient()


class MockEmbeddings(Embeddings):
    """Deterministic, dependency-free embedding for tests.

    Hashes each token into a fixed-dimension bag-of-words vector, so texts that
    share vocabulary land closer in cosine space. Not semantically meaningful,
    but stable and good enough to exercise the scoring/ranking path.
    """

    dim: int = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _get_vllm_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=_require_env("EMBEDDING_MODEL_NAME"),
        base_url=_require_env("EMBEDDING_BASE_URL"),
        api_key=SecretStr(_require_env("EMBEDDING_API_KEY")),
    )


def get_embeddings_client() -> Embeddings:
    backend = os.getenv("EMBEDDING_BACKEND", "mock")
    if backend == "vllm":
        return _get_vllm_embeddings()
    return MockEmbeddings()