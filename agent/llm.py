import hashlib
import json
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class MockLLMClient(BaseChatModel):
    @property
    def llm_type(self) -> str:
        return "mock"
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        fixture_path = Path("/tests/fixtures/mock_responses.json")
        responses = json.loads(fixture_path.read_text())
        content = responses[0]["content"]
        response = AIMessage(content=content)
        return ChatResult(generation=[ChatGeneration(message=response)])
    
def _get_vllm_client() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("VLLM_MODEL_NAME"),
        base_url=os.getenv("VllM_BASE_URL"),
        api_key=os.getenv("VLLM_API_KEY"),
        temperature=float(os.getenv("VLLM_TEMPERATURE")),
        max_tokens=int(os.getenv("VLLM_MAX_TOKENS"))
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
        model=os.getenv("EMBEDDING_MODEL_NAME"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
    )


def get_embeddings_client() -> Embeddings:
    backend = os.getenv("EMBEDDING_BACKEND", "mock")
    if backend == "vllm":
        return _get_vllm_embeddings()
    return MockEmbeddings()