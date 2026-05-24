"""
providers/anthropic_provider.py
────────────────────────────────
HTTP adapter for the **Anthropic** Messages API
(https://docs.anthropic.com/en/api/messages).

Key differences vs. the OpenAI adapter
───────────────────────────────────────
• The Anthropic API separates ``system`` messages from the ``messages`` list.
  The first system-role message (if any) is extracted and sent as the top-level
  ``system`` field; the rest become the ``messages`` array.
• Authentication uses the ``x-api-key`` header (not ``Authorization: Bearer``).
• The API version header ``anthropic-version`` is required.
• Response structure differs: ``content[].text`` vs. ``choices[].message.content``.

All provider modules expose the same interface::

    response: ChatCompletionResponse = await complete(request)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

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

# Anthropic API version to pin (update as new stable versions are released)
ANTHROPIC_API_VERSION = "2023-06-01"

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP client
# ─────────────────────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
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


def _split_system_messages(
    messages: List[ChatMessage],
) -> Tuple[Optional[str], List[ChatMessage]]:
    """
    Separate the first system-role message (Anthropic's ``system`` field)
    from the rest of the conversation turns.

    Returns
    -------
    system_prompt:
        Content of the first system message, or ``None`` if absent.
    conversation:
        Remaining messages (may still contain further system messages which
        are left in place – Anthropic supports them inline in newer versions).
    """
    system_prompt: Optional[str] = None
    conversation: List[ChatMessage] = []

    for msg in messages:
        if msg.role == Role.system and system_prompt is None:
            # Only the *first* system message is hoisted
            system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
        else:
            conversation.append(msg)

    return system_prompt, conversation


def _build_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Translate ``ChatCompletionRequest`` → Anthropic Messages API payload.

    Reference: https://docs.anthropic.com/en/api/messages
    """
    model = request.model or settings.anthropic_default_model
    system_prompt, conversation = _split_system_messages(request.messages)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": msg.role.value, "content": msg.content}
            for msg in conversation
        ],
        # max_tokens is *required* by the Anthropic API
        "max_tokens": request.max_tokens or 4096,
    }

    if system_prompt:
        payload["system"] = system_prompt
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop is not None:
        payload["stop_sequences"] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )

    payload.update(request.extra_params)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

# Map Anthropic stop-reason strings to the internal FinishReason enum
_STOP_REASON_MAP: Dict[str, FinishReason] = {
    "end_turn": FinishReason.stop,
    "max_tokens": FinishReason.length,
    "stop_sequence": FinishReason.stop,
    "tool_use": FinishReason.tool_calls,
}


def _parse_response(data: Dict[str, Any]) -> ChatCompletionResponse:
    """Parse the raw Anthropic API response into a ``ChatCompletionResponse``."""
    # Anthropic returns content as a list of blocks; join text blocks.
    content_blocks = data.get("content", [])
    text = " ".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )

    stop_reason = data.get("stop_reason")
    finish_reason = _STOP_REASON_MAP.get(stop_reason) if stop_reason else None

    usage_data = data.get("usage", {})

    return ChatCompletionResponse(
        id=data.get("id", f"anthropic-{uuid.uuid4().hex}"),
        model=data.get("model", settings.anthropic_default_model),
        provider=ProviderName.anthropic,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role=Role.assistant, content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageStats(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0)
            + usage_data.get("output_tokens", 0),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


@with_retry()
async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Send a chat-completion request to the Anthropic Messages API.

    Parameters
    ----------
    request:
        Validated ``ChatCompletionRequest``.

    Returns
    -------
    ChatCompletionResponse
        Normalised response.

    Raises
    ------
    httpx.HTTPStatusError
        For non-retryable 4xx errors.
    """
    payload = _build_payload(request)
    logger.debug("Anthropic request: model=%s", payload.get("model"))

    client = _get_client()
    response = await client.post("/v1/messages", json=payload)
    response.raise_for_status()

    return _parse_response(response.json())
