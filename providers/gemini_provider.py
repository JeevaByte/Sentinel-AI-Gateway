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


class GeminiProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = getattr(config, "api_key", "")
        self.timeout = float(getattr(config, "timeout", 30))
        self.base_url = getattr(config, "base_url", "https://generativelanguage.googleapis.com")

    async def complete(self, prompt: str) -> str:
        endpoint = (
            f"{self.base_url.rstrip('/')}/v1beta/models/gemini-1.5-flash:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
            return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    async def health_check(self) -> None:
        endpoint = (
            f"{self.base_url.rstrip('/')}/v1beta/models/gemini-1.5-flash:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    @staticmethod
    def _format_error(exc: Exception) -> str:
        details = ""
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                body: dict[str, Any] = exc.response.json()
                details = body.get("error", {}).get("status") or body.get("error", {}).get("message", "")
            except Exception:
                details = exc.response.text[:200]
        detail_str = f" - {details}" if details else ""
        return f"{type(exc).__name__}{detail_str}: {exc}"
