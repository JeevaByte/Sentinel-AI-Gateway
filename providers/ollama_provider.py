"""
providers/ollama_provider.py
─────────────────────────────
HTTP adapter for **Ollama** – a local LLM runtime
(https://ollama.com / https://github.com/ollama/ollama).

Key differences vs. cloud providers
────────────────────────────────────
• No authentication required (Ollama runs locally by default).
• Two API variants are supported:
    - ``/api/chat``  – native Ollama chat format (preferred, v0.1.14+)
    - ``/v1/chat/completions`` – OpenAI-compatible shim (available since
      Ollama v0.1.24; enabled automatically when present)
  This module uses the native ``/api/chat`` endpoint and is therefore not
  dependent on the OpenAI-compatibility layer being enabled.
• The response structure differs: ``message.content`` instead of
  ``choices[].message.content``.
• No usage statistics in the base response (some fields are approximated).

All provider modules expose the same interface::

    response: ChatCompletionResponse = await complete(request)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

import httpx

from config import settings
from models.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FinishReason,
    ProviderName,
    Role,
    UsageStats,
)
from reliability.retry import with_retry

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP client
# ─────────────────────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            headers={"Content-Type": "application/json"},
            # Ollama can be slow on first token for large models – use a
            # generous timeout while still surfacing hangs.
            timeout=httpx.Timeout(max(settings.request_timeout, 120.0)),
        )
    return _client


async def close_client() -> None:
    """Close the shared HTTP client.  Call from FastAPI ``shutdown`` lifespan."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# ─────────────────────────────────────────────────────────────────────────────
# Request translation
# ─────────────────────────────────────────────────────────────────────────────


def _build_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Translate ``ChatCompletionRequest`` → Ollama ``/api/chat`` payload.

    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    """
    model = request.model or settings.ollama_default_model

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ],
        # stream=False so we get a single JSON response (not NDJSON lines)
        "stream": False,
    }

    # Ollama passes generation params inside an ``options`` sub-object
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.stop is not None:
        options["stop"] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )

    if options:
        payload["options"] = options

    payload.update(request.extra_params)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

# Map Ollama done_reason to internal FinishReason
_DONE_REASON_MAP: Dict[str, FinishReason] = {
    "stop": FinishReason.stop,
    "length": FinishReason.length,
    "load": FinishReason.stop,  # model was just loaded; no generation error
}


def _parse_response(data: Dict[str, Any]) -> ChatCompletionResponse:
    """Parse an Ollama ``/api/chat`` non-streaming response."""
    msg_data = data.get("message", {})
    done_reason = data.get("done_reason")
    finish = _DONE_REASON_MAP.get(done_reason) if done_reason else None

    # Ollama provides token counts for completed responses
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return ChatCompletionResponse(
        id=f"ollama-{uuid.uuid4().hex}",
        model=data.get("model", settings.ollama_default_model),
        provider=ProviderName.ollama,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role=Role(msg_data.get("role", "assistant")),
                    content=msg_data.get("content", ""),
                ),
                finish_reason=finish,
            )
        ],
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


@with_retry()
async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Send a chat-completion request to the local Ollama runtime.

    Parameters
    ----------
    request:
        Validated ``ChatCompletionRequest``.

    Returns
    -------
    ChatCompletionResponse
        Normalised response.

    Notes
    -----
    Ollama must be running locally (or at ``settings.ollama_base_url``) and
    the requested model must already be pulled (``ollama pull <model>``).
    If the daemon is not reachable, ``httpx.NetworkError`` is raised and the
    circuit breaker will count the failure.
    """
    payload = _build_payload(request)
    logger.debug("Ollama request: model=%s", payload.get("model"))

    client = _get_client()
    response = await client.post("/api/chat", json=payload)
    response.raise_for_status()

    return _parse_response(response.json())
