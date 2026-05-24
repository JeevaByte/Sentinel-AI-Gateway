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
        base_url: str


class CrusoeProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = getattr(config, "api_key", "")
        self.timeout = float(getattr(config, "timeout", 30))
        # Open-source fallback provider running on Crusoe Cloud via OpenAI-compatible inference API.
        self.base_url = getattr(config, "base_url", "https://inference.crusoe.ai/v1")

    async def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    async def health_check(self) -> None:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
                    or ""
                )
            except Exception:
                details = exc.response.text[:200]
        detail_str = f" - {details}" if details else ""
        return f"{type(exc).__name__}{detail_str}: {exc}"
