from __future__ import annotations

from typing import Any

import httpx

try:
    from .base import ProviderConfig, ProviderException
except ImportError:  # pragma: no cover - compatibility fallback
    class ProviderException(Exception):
        pass

    class ProviderConfig:  # type: ignore[override]
        api_key: str
        timeout: float


class AnthropicProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = getattr(config, "api_key", "")
        self.timeout = float(getattr(config, "timeout", 30))
        self.base_url = getattr(config, "base_url", "https://api.anthropic.com/v1")

    async def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/messages"
        payload = {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            return str(data["content"][0]["text"]).strip()
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    async def health_check(self) -> None:
        url = f"{self.base_url.rstrip('/')}/messages"
        payload = {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    @staticmethod
    def _format_error(exc: Exception) -> str:
        details = ""
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                body: dict[str, Any] = exc.response.json()
                details = (
                    body.get("error", {}).get("type")
                    or body.get("error", {}).get("message")
                    or body.get("type")
                    or body.get("message", "")
                )
            except Exception:
                details = exc.response.text[:200]
        detail_str = f" - {details}" if details else ""
        return f"{type(exc).__name__}{detail_str}: {exc}"
