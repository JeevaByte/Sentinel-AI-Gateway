"""
models/__init__.py
──────────────────
Package marker.  Re-exports the public schema types for convenient imports:

    from models import ChatCompletionRequest, ChatCompletionResponse
"""

from models.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FinishReason,
    GatewayError,
    GatewayHealthResponse,
    ProviderName,
    ProviderStatus,
    Role,
    UsageStats,
)

__all__ = [
    "ChatChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "FinishReason",
    "GatewayError",
    "GatewayHealthResponse",
    "ProviderName",
    "ProviderStatus",
    "Role",
    "UsageStats",
]
