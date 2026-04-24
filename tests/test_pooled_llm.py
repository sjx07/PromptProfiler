"""Pooled LLM wrapper behavior."""
from __future__ import annotations

import inspect
import queue
from types import SimpleNamespace

from execution.pooled_llm import PooledLLMCall


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def test_pooled_llm_default_max_tokens_is_2048():
    signature = inspect.signature(PooledLLMCall)

    assert signature.parameters["max_tokens"].default == 2048


def test_pooled_llm_uses_configured_max_tokens():
    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    llm = PooledLLMCall.__new__(PooledLLMCall)
    llm._model = "fake-model"
    llm._max_tokens = 2048
    llm._ext_client = fake_client
    llm._clients = None
    llm._port_pool = queue.Queue()
    llm._port_pool.put(0)

    result = llm("system", "user")

    assert result["raw_response"] == "ok"
    assert completions.kwargs["max_tokens"] == 2048
