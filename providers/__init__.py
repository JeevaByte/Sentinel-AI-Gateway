"""
providers/__init__.py
──────────────────────
Package marker for the providers layer.

Exposes a ``PROVIDER_MAP`` dict that maps provider-name strings to their
``complete`` coroutines.  The router imports this map at startup to register
every known provider without having to hard-code provider names elsewhere.

Usage in main.py::

    from providers import PROVIDER_MAP

    for name, fn in PROVIDER_MAP.items():
        router.register(name, fn)
"""

from __future__ import annotations

from typing import Callable, Coroutine, Any, Dict

from models.schemas import ChatCompletionRequest, ChatCompletionResponse

# Import each provider's ``complete`` coroutine
from providers import (
    anthropic_provider,
    crusoe_provider,
    gemini_provider,
    ollama_provider,
    openai_provider,
)

# Type alias for a provider callable
ProviderFn = Callable[
    [ChatCompletionRequest], Coroutine[Any, Any, ChatCompletionResponse]
]

#: Maps provider name → async complete() coroutine.
#: Add new providers here to make them available to the router automatically.
PROVIDER_MAP: Dict[str, ProviderFn] = {
    "openai": openai_provider.complete,
    "anthropic": anthropic_provider.complete,
    "gemini": gemini_provider.complete,
    "crusoe": crusoe_provider.complete,
    "ollama": ollama_provider.complete,
}

#: ``close_client`` callables for clean shutdown (called in lifespan).
PROVIDER_SHUTDOWN_HOOKS = [
    openai_provider.close_client,
    anthropic_provider.close_client,
    gemini_provider.close_client,
    crusoe_provider.close_client,
    ollama_provider.close_client,
]

__all__ = ["PROVIDER_MAP", "PROVIDER_SHUTDOWN_HOOKS"]
