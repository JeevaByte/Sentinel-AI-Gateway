from __future__ import annotations

from typing import Any

import httpx

try:
    from .base import ProviderConfig, ProviderException
except ImportError:  # pragma: no cover - compatibility fallback
    class ProviderException(Exception):
        pass

    class ProviderConfig:  # type: ignore[override]
        timeout: float
        base_url: str


class OllamaProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.timeout = float(getattr(config, "timeout", 30))
        # Last-resort local fallback; requires Ollama running locally.
        self.base_url = getattr(config, "base_url", "http://localhost:11434")

    async def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            return str(data["response"]).strip()
        except httpx.ConnectError as exc:
            message = str(exc).lower()
            if "connection refused" in message or "actively refused" in message:
                raise ProviderException(
                    f"{type(exc).__name__}: Ollama is not running on localhost:11434"
                ) from exc
            raise ProviderException(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    async def health_check(self) -> None:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 1},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            message = str(exc).lower()
            if "connection refused" in message or "actively refused" in message:
                raise ProviderException(
                    f"{type(exc).__name__}: Ollama is not running on localhost:11434"
                ) from exc
            raise ProviderException(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise ProviderException(self._format_error(exc)) from exc

    @staticmethod
    def _format_error(exc: Exception) -> str:
        details = ""
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                body: dict[str, Any] = exc.response.json()
                details = body.get("error", "") or body.get("message", "")
            except Exception:
                details = exc.response.text[:200]
        detail_str = f" - {details}" if details else ""
        return f"{type(exc).__name__}{detail_str}: {exc}"
