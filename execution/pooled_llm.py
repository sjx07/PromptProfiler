"""Port-pooled LLM call with AutoVLLM instrumentation.

Supports two modes:
- Local vLLM (default): port-pooled AutoVLLM clients at 127.0.0.1:{port}/v1
- External API: single openai.OpenAI client with a custom base_url (DeepSeek, OpenAI, etc.)
"""
from __future__ import annotations

import json
import os
import queue
from typing import Any, Dict, List, Optional

from autovllm import AutoVLLM
from autovllm.store import TrajectoryStore as VLLMStore

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]


class PooledLLMCall:
    """Port-pooled llm_call with AutoVLLM instrumentation."""

    def __init__(
        self,
        model: str,
        ports: List[int],
        slots_per_port: int = 20,
        *,
        labels: Dict[str, Any] | None = None,
        vllm_store: VLLMStore | None = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        request_timeout: int = 240,
    ) -> None:

        self._model = model
        self._max_tokens = int(max_tokens)
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        self._ext_client: Optional[Any] = None
        self._clients: Optional[Dict[int, AutoVLLM]] = None

        if base_url is not None:
            # External API mode (DeepSeek, OpenAI, etc.) — single client, no AutoVLLM
            if _openai is None:
                raise ImportError("openai package required for external API mode: pip install openai")
            _key = api_key or os.environ.get("OPENAI_API_KEY", "no-key")
            self._ext_client = _openai.OpenAI(base_url=base_url, api_key=_key, timeout=request_timeout)
        else:
            # Local vLLM port-pooling mode
            self._clients = {
                p: AutoVLLM(
                    base_url=f"http://127.0.0.1:{p}/v1",
                    model=model,
                    ports=ports,
                    labels=labels or {},
                    store=vllm_store,
                    timeout=request_timeout,
                )
                for p in ports
            }
            # All clients share one session
            session_id = self._clients[ports[0]].session_id
            for p in ports[1:]:
                self._clients[p]._session_id = session_id

        self._port_pool: queue.Queue[int] = queue.Queue()
        for p in ports:
            for _ in range(slots_per_port):
                self._port_pool.put(p)

    def __call__(self, system_prompt: str, user_content: str) -> dict:
        port = self._port_pool.get()
        try:
            client = self._ext_client if self._ext_client is not None else self._clients[port]  # type: ignore[index]
            request: Dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": self._max_tokens,
            }
            if self._temperature is not None:
                request["temperature"] = self._temperature
            if self._top_p is not None:
                request["top_p"] = self._top_p
            if self._top_k is not None:
                request["extra_body"] = {"top_k": self._top_k}

            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            message = choice.message
            content = _get_attr_or_key(message, "content")
            reasoning = (
                _get_attr_or_key(message, "reasoning_content")
                or _get_attr_or_key(message, "reasoning")
            )
            raw = _coerce_optional_text(content)
            usage = getattr(response, "usage", None)
            return {
                "raw_response": raw,
                "raw_reasoning": _coerce_optional_text(reasoning),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "finish_reason": _get_attr_or_key(choice, "finish_reason"),
            }
        finally:
            self._port_pool.put(port)

    def set_labels(self, **kwargs: Any) -> None:
        if self._clients is not None:
            for client in self._clients.values():
                client.set_labels(**kwargs)

    def close(self) -> None:
        if self._clients is not None:
            for client in self._clients.values():
                client.close()


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _coerce_optional_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value:
        return json.dumps(value)
    return ""
