"""
providers/gemini_provider.py
─────────────────────────────
HTTP adapter for the **Google Gemini** GenerateContent API
(https://ai.google.dev/api/generate-content).

Key differences vs. OpenAI adapter
────────────────────────────────────
• Authentication uses a query-parameter API key (``?key=<GEMINI_API_KEY>``),
  not a Bearer token header.
• The request body uses ``contents[].parts[].text`` instead of
  ``messages[].content``.
• Role mapping: OpenAI ``user`` → Gemini ``user``;
                OpenAI ``assistant`` → Gemini ``model``.
• System instructions are passed as a separate top-level field
  ``systemInstruction``.
• Response structure: ``candidates[].content.parts[].text``.
• Model endpoint pattern: ``/models/{model}:generateContent``.

All provider modules expose the same interface::

    response: ChatCompletionResponse = await complete(request)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

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
# Role mapping: internal → Gemini
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_MAP: Dict[str, str] = {
    Role.user.value: "user",
    Role.assistant.value: "model",
    # system is handled separately as systemInstruction
    Role.system.value: "user",  # fallback if not extracted
}

# Map Gemini finishReason strings to internal FinishReason enum
_FINISH_REASON_MAP: Dict[str, FinishReason] = {
    "STOP": FinishReason.stop,
    "MAX_TOKENS": FinishReason.length,
    "SAFETY": FinishReason.content_filter,
    "RECITATION": FinishReason.content_filter,
    "OTHER": FinishReason.stop,
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP client
# ─────────────────────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.gemini_base_url,
            headers={"Content-Type": "application/json"},
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


def _extract_system_instruction(
    messages: List[ChatMessage],
) -> tuple[Optional[Dict[str, Any]], List[ChatMessage]]:
    """
    Pull the first system-role message out and format it as a Gemini
    ``systemInstruction`` object.  Returns ``(instruction, remaining_msgs)``.
    """
    system_instruction: Optional[Dict[str, Any]] = None
    remaining: List[ChatMessage] = []

    for msg in messages:
        if msg.role == Role.system and system_instruction is None:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            system_instruction = {"parts": [{"text": text}]}
        else:
            remaining.append(msg)

    return system_instruction, remaining


def _build_contents(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert internal ``ChatMessage`` list to Gemini ``contents`` array.

    Each content entry: ``{"role": "user"|"model", "parts": [{"text": "..."}]}``.
    """
    contents = []
    for msg in messages:
        role = _ROLE_MAP.get(msg.role.value, "user")
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


def _build_payload(request: ChatCompletionRequest) -> tuple[str, Dict[str, Any]]:
    """
    Build the Gemini request payload and return ``(model_name, payload)``.

    The model name is returned separately because it forms part of the URL path.
    """
    model = request.model or settings.gemini_default_model
    system_instruction, conversation = _extract_system_instruction(request.messages)

    generation_config: Dict[str, Any] = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["topP"] = request.top_p
    if request.max_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_tokens
    if request.stop is not None:
        generation_config["stopSequences"] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )

    payload: Dict[str, Any] = {
        "contents": _build_contents(conversation),
    }

    if generation_config:
        payload["generationConfig"] = generation_config
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    payload.update(request.extra_params)
    return model, payload


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_response(data: Dict[str, Any], model: str) -> ChatCompletionResponse:
    """Parse Gemini ``generateContent`` response into a ``ChatCompletionResponse``."""
    candidates = data.get("candidates", [])
    choices: List[ChatChoice] = []

    for idx, candidate in enumerate(candidates):
        parts = candidate.get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if "text" in p)
        raw_finish = candidate.get("finishReason")
        finish = _FINISH_REASON_MAP.get(raw_finish) if raw_finish else None

        choices.append(
            ChatChoice(
                index=idx,
                message=ChatMessage(role=Role.assistant, content=text),
                finish_reason=finish,
            )
        )

    usage_meta = data.get("usageMetadata", {})
    return ChatCompletionResponse(
        id=f"gemini-{uuid.uuid4().hex}",
        model=model,
        provider=ProviderName.gemini,
        choices=choices,
        usage=UsageStats(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


@with_retry()
async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Send a chat-completion request to the Google Gemini API.

    Parameters
    ----------
    request:
        Validated ``ChatCompletionRequest``.

    Returns
    -------
    ChatCompletionResponse
        Normalised response.
    """
    model, payload = _build_payload(request)
    endpoint = f"/models/{model}:generateContent"
    params = {"key": settings.gemini_api_key}

    logger.debug("Gemini request: model=%s endpoint=%s", model, endpoint)

    client = _get_client()
    response = await client.post(endpoint, json=payload, params=params)
    response.raise_for_status()

    return _parse_response(response.json(), model)
