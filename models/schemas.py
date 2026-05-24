"""
models/schemas.py
─────────────────
Pydantic models shared across the whole application.

These data-classes define the public API contract (request / response bodies)
and the internal data structures passed between the router, providers, and
monitoring layers.

Design notes:
  • Use pydantic v2 model syntax (model_config, field validators).
  • Keep *request* models separate from *response* models – they evolve
    independently.
  • The ``ChatMessage`` model mirrors the OpenAI chat-completion message
    schema so that every provider adapter only needs to translate *to* its
    own wire format, not to a bespoke internal format.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class Role(str, Enum):
    """Valid roles for a chat message, following the OpenAI convention."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ProviderName(str, Enum):
    """Identifiers for all supported LLM back-ends."""

    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"
    crusoe = "crusoe"
    ollama = "ollama"


class FinishReason(str, Enum):
    """Why the model stopped generating tokens."""

    stop = "stop"
    length = "length"
    tool_calls = "tool_calls"
    content_filter = "content_filter"
    error = "error"


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """
    A single turn in a conversation.

    ``content`` may be:
      - a plain string for simple text messages
      - a list of content-part dicts (for multi-modal inputs / tool results)
    """

    role: Role
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None  # For tool / function messages
    tool_call_id: Optional[str] = None  # For role=tool messages

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: object) -> object:
        """Ensure content is not an empty string or empty list."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("content must not be empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("content list must not be empty")
        return v


class UsageStats(BaseModel):
    """Token usage reported by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    """
    Body for ``POST /v1/chat/completions``.

    Mirrors the OpenAI chat-completion API so existing clients work without
    modification.  Extra fields are forwarded verbatim to whichever provider
    is selected.
    """

    # Required
    messages: List[ChatMessage] = Field(..., min_length=1)

    # Optional – falls back to the provider's configured default model
    model: Optional[str] = None

    # Provider selection hints
    provider: Optional[ProviderName] = Field(
        default=None,
        description=(
            "Pin the request to a specific provider. "
            "If omitted, the router picks one according to priority / health."
        ),
    )

    # Generation parameters (forwarded to the provider as-is when present)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

    # Catch-all for provider-specific extras (e.g. Anthropic's system prompt)
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific parameters passed through unchanged.",
    )

    @model_validator(mode="after")
    def validate_stream_not_implemented(self) -> "ChatCompletionRequest":
        """Streaming is not yet supported in v0.1 – reject early."""
        # TODO: Remove this guard once streaming SSE support is added.
        if self.stream:
            raise ValueError(
                "Streaming responses are not yet supported. Set stream=false."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────


class ChatChoice(BaseModel):
    """One completion choice returned by the provider."""

    index: int = 0
    message: ChatMessage
    finish_reason: Optional[FinishReason] = None


class ChatCompletionResponse(BaseModel):
    """
    Unified response body returned by ``POST /v1/chat/completions``.

    Always normalised to this shape regardless of which provider was used,
    so the caller never needs to know.
    """

    id: str  # Provider-assigned completion ID
    object: str = "chat.completion"
    model: str  # Actual model name used (may differ from request)
    provider: ProviderName  # Which back-end handled the request
    choices: List[ChatChoice]
    usage: UsageStats = Field(default_factory=UsageStats)

    # Gateway-level metadata (timing, retries, circuit-breaker state)
    gateway_meta: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Health / status models
# ─────────────────────────────────────────────────────────────────────────────


class ProviderStatus(BaseModel):
    """Health and circuit-breaker status for a single provider."""

    name: ProviderName
    healthy: bool
    circuit_state: str  # "closed" | "open" | "half_open"
    failure_count: int = 0
    last_failure_at: Optional[str] = None  # ISO-8601 timestamp


class GatewayHealthResponse(BaseModel):
    """
    Response body for ``GET /health``.

    Reports the overall gateway state and the per-provider breakdown.
    """

    status: str  # "ok" | "degraded" | "down"
    providers: List[ProviderStatus]
    version: str = "0.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Error models
# ─────────────────────────────────────────────────────────────────────────────


class GatewayError(BaseModel):
    """
    Standardised error envelope returned on 4xx / 5xx responses.

    The ``provider_errors`` list records every provider that was tried and
    what error it returned, which is useful for debugging routing decisions.
    """

    error: str  # Human-readable summary
    code: str  # Machine-readable error code, e.g. "all_providers_failed"
    provider_errors: List[Dict[str, Any]] = Field(default_factory=list)
