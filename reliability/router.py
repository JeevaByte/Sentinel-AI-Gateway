"""
reliability/router.py

Provider selection with circuit-breaker logic.

This module exposes:
  - ProviderRouter   – selects the best available provider and executes
                       the chat request with automatic fallback.
  - get_provider_statuses – returns circuit-breaker state for every provider.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import httpx

from config import ProviderConfig, load_provider_configs
from models.schemas import ChatResponse, ProviderStatus


class CircuitState(str, Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing – requests skipped
    HALF_OPEN = "half_open" # probe request allowed


# seconds a circuit stays OPEN before moving to HALF_OPEN
_OPEN_DURATION = 60
# consecutive failures required to trip the circuit
_FAILURE_THRESHOLD = 3


class _CircuitBreaker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure: Optional[datetime] = None
        self._opened_at: Optional[float] = None

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure = datetime.utcnow()
        if self.failure_count >= _FAILURE_THRESHOLD:
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self._opened_at and (time.monotonic() - self._opened_at) >= _OPEN_DURATION:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN – allow one probe
        return True

    def to_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            state=self.state.value,
            failure_count=self.failure_count,
            last_failure=self.last_failure,
        )


# Module-level registry so state persists across requests
_breakers: Dict[str, _CircuitBreaker] = {}
_providers: List[ProviderConfig] = []


def _init() -> None:
    global _providers, _breakers
    if not _providers:
        _providers = load_provider_configs()
        _breakers = {p.name: _CircuitBreaker(p.name) for p in _providers}


async def _call_provider(provider: ProviderConfig, prompt: str) -> str:
    """Send a chat completion request to the given provider and return the text."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    payload: Dict = {}

    name = provider.name

    if name == "openai":
        headers["Authorization"] = f"Bearer {provider.api_key}"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        url = f"{provider.base_url}/chat/completions"
    elif name == "anthropic":
        headers["x-api-key"] = provider.api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        url = f"{provider.base_url}/messages"
    elif name == "gemini":
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        url = (
            f"{provider.base_url}/models/gemini-1.5-flash:generateContent"
            f"?key={provider.api_key}"
        )
    elif name == "crusoe":
        headers["Authorization"] = f"Bearer {provider.api_key}"
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
        }
        url = f"{provider.base_url}/chat/completions"
    elif name == "ollama":
        payload = {"model": "llama3", "prompt": prompt, "stream": False}
        url = f"{provider.base_url}/api/generate"
    else:
        raise ValueError(f"Unknown provider: {name}")

    async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Extract text from provider-specific response shape
    if name == "openai" or name == "crusoe":
        return data["choices"][0]["message"]["content"]
    elif name == "anthropic":
        return data["content"][0]["text"]
    elif name == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"]
    elif name == "ollama":
        return data["response"]
    return ""


class ProviderRouter:
    """Stateless helper – all state lives in module-level _breakers."""

    def __init__(self) -> None:
        _init()

    async def chat(self, prompt: str, user_id: str) -> ChatResponse:
        start = time.monotonic()
        attempt_count = 0
        fallback_triggered = False
        last_error: Optional[Exception] = None

        for provider in _providers:
            breaker = _breakers[provider.name]
            if not breaker.is_available():
                continue

            if attempt_count > 0:
                fallback_triggered = True
            attempt_count += 1

            try:
                text = await _call_provider(provider, prompt)
                breaker.record_success()
                latency_ms = (time.monotonic() - start) * 1000
                return ChatResponse(
                    response=text,
                    provider_used=provider.name,
                    latency_ms=round(latency_ms, 2),
                    fallback_triggered=fallback_triggered,
                    attempt_count=attempt_count,
                )
            except Exception as exc:  # noqa: BLE001
                breaker.record_failure()
                last_error = exc

        raise RuntimeError(
            f"All providers exhausted after {attempt_count} attempt(s). "
            f"Last error: {last_error}"
        )


def get_provider_statuses() -> List[ProviderStatus]:
    _init()
    return [b.to_status() for b in _breakers.values()]
