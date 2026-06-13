# agent/llm.py
import json
import os
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


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
        model=os.getenv("VLLM_MODEL_NAME", "mistralal/Mistral-7B-Instruct-v0.3"),
        base_url=os.getenv("VllM_BASE_URL"),
        api_key=os.getenv("VLLM_API_KEY"),
        temperature=float(os.getenv("VLLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "1000"))
    )



def get_llm_client() -> BaseChatModel:
    backend = os.getenv("LLM_BACKEND", "mock")
    if backend == "vllm":
        return _get_vllm_client()
    return MockLLMClient()