"""
providers/crusoe_provider.py
─────────────────────────────
HTTP adapter for the **Crusoe Cloud** inference API.

Crusoe exposes an OpenAI-compatible Chat Completions endpoint, so this
adapter is intentionally thin – it reuses the same payload builder and
response parser as the OpenAI provider, only swapping the base URL and
authentication header.

Reference: https://docs.crusoe.ai/inference/

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
            base_url=settings.crusoe_base_url,
            headers={
                "Authorization": f"Bearer {settings.crusoe_api_key}",
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
# Request translation  (OpenAI-compatible payload)
# ─────────────────────────────────────────────────────────────────────────────


def _build_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Build an OpenAI-compatible JSON payload for the Crusoe inference API.

    Crusoe follows the OpenAI Chat Completions schema, so this is nearly
    identical to the OpenAI provider's payload builder.
    """
    model = request.model or settings.crusoe_default_model

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

    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.stop is not None:
        payload["stop"] = request.stop

    payload.update(request.extra_params)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing  (OpenAI-compatible response)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_response(data: Dict[str, Any]) -> ChatCompletionResponse:
    """Parse the Crusoe API response (OpenAI-schema) into a ``ChatCompletionResponse``."""
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
        id=data.get("id", f"crusoe-{uuid.uuid4().hex}"),
        model=data.get("model", settings.crusoe_default_model),
        provider=ProviderName.crusoe,
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
    Send a chat-completion request to the Crusoe Cloud inference API.

    Parameters
    ----------
    request:
        Validated ``ChatCompletionRequest``.

    Returns
    -------
    ChatCompletionResponse
        Normalised response.
    """
    payload = _build_payload(request)
    logger.debug("Crusoe request: model=%s", payload.get("model"))

    client = _get_client()
    response = await client.post("/chat/completions", json=payload)
    response.raise_for_status()

    return _parse_response(response.json())
