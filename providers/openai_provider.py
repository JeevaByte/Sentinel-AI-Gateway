"""
providers/openai_provider.py
─────────────────────────────
HTTP adapter for the **OpenAI** Chat Completions API
(https://platform.openai.com/docs/api-reference/chat).

This module exposes a single async ``complete`` coroutine that:
  1. Translates the internal ``ChatCompletionRequest`` into the OpenAI
     wire format.
  2. Sends the request via a shared ``httpx.AsyncClient``.
  3. Parses the response JSON into a ``ChatCompletionResponse``.
  4. Applies the ``@with_retry`` decorator for transient-error resilience.

All provider modules follow the same interface so the router can call them
interchangeably::

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
# Shared HTTP client (created once, reused across requests)
# ─────────────────────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """
    Return (or lazily create) the shared async HTTP client for OpenAI.

    Call ``close_client()`` during application shutdown to cleanly drain
    any in-flight connections.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.request_timeout),
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
    Translate a ``ChatCompletionRequest`` into the OpenAI JSON payload.

    The OpenAI chat-completion schema is used as the canonical internal
    format, so this translation is mostly a pass-through.  Extra params
    from ``request.extra_params`` are merged in last, allowing callers to
    pass OpenAI-specific fields like ``response_format`` or ``tools``.
    """
    model = request.model or settings.openai_default_model

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": msg.role.value,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
            }
            for msg in request.messages
        ],
    }

    # Optional generation parameters – only include if caller specified them
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.stop is not None:
        payload["stop"] = request.stop

    # Merge provider-specific extras (caller's intent wins)
    payload.update(request.extra_params)

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_response(data: Dict[str, Any]) -> ChatCompletionResponse:
    """
    Parse the raw OpenAI API response dict into a ``ChatCompletionResponse``.

    Handles the ``choices[].message`` → ``ChatMessage`` mapping and maps
    OpenAI finish-reason strings to the ``FinishReason`` enum.
    """
    usage_data = data.get("usage", {})

    choices = []
    for idx, choice in enumerate(data.get("choices", [])):
        msg_data = choice.get("message", {})
        finish = choice.get("finish_reason")
        choices.append(
            ChatChoice(
                index=idx,
                message=ChatMessage(
                    role=Role(msg_data.get("role", "assistant")),
                    content=msg_data.get("content", ""),
                ),
                finish_reason=FinishReason(finish) if finish else None,
            )
        )

    return ChatCompletionResponse(
        id=data.get("id", f"openai-{uuid.uuid4().hex}"),
        model=data.get("model", settings.openai_default_model),
        provider=ProviderName.openai,
        choices=choices,
        usage=UsageStats(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


@with_retry()
async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Send a chat-completion request to the OpenAI API.

    Parameters
    ----------
    request:
        Validated ``ChatCompletionRequest`` from the gateway endpoint.

    Returns
    -------
    ChatCompletionResponse
        Normalised response ready for the caller.

    Raises
    ------
    httpx.HTTPStatusError
        For non-retryable 4xx errors (e.g. 401 Unauthorized, 400 Bad Request).
    httpx.TimeoutException / httpx.NetworkError
        For transient network failures (will be retried automatically).
    """
    payload = _build_payload(request)
    logger.debug("OpenAI request payload: model=%s", payload.get("model"))

    client = _get_client()
    response = await client.post("/chat/completions", json=payload)

    # Raise for 4xx / 5xx – tenacity will decide whether to retry
    response.raise_for_status()

    data: Dict[str, Any] = response.json()
    return _parse_response(data)
