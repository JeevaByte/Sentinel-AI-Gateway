"""
reliability/router.py
──────────────────────
Smart request router that selects which LLM provider should handle each
incoming chat-completion request.

Responsibilities
────────────────
1. Maintain one ``CircuitBreaker`` instance per registered provider.
2. On each request, iterate through the *priority list* and pick the first
   provider whose circuit is not OPEN.
3. If a caller pins a specific provider (``request.provider``), honour that
   choice but still enforce circuit-breaker gating.
4. After a provider call:
     - on success  → ``cb.record_success()``
     - on failure  → ``cb.record_failure()``
5. If every provider fails, raise ``AllProvidersFailedError``.

Usage::

    from reliability.router import Router

    router = Router()

    # Register provider callables once at startup
    router.register("openai", openai_provider.complete)
    router.register("anthropic", anthropic_provider.complete)
    ...

    # Later, in the endpoint handler:
    response = await router.route(request)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from config import settings
from models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ProviderName,
    ProviderStatus,
)
from reliability.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────


class AllProvidersFailedError(Exception):
    """
    Raised when every provider in the priority list has been tried and all
    have either returned an error or been rejected by an open circuit breaker.

    The ``provider_errors`` dict maps provider name → last exception.
    """

    def __init__(self, provider_errors: Dict[str, Exception]) -> None:
        names = ", ".join(provider_errors)
        super().__init__(f"All providers failed: {names}")
        self.provider_errors = provider_errors


# ─────────────────────────────────────────────────────────────────────────────
# Provider callable type alias
# ─────────────────────────────────────────────────────────────────────────────

#: Type of the async callable each provider module must expose.
ProviderCallable = Callable[
    [ChatCompletionRequest], Coroutine[Any, Any, ChatCompletionResponse]
]


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────


class Router:
    """
    Priority-based, circuit-breaker-aware request router.

    Parameters
    ----------
    priority:
        Ordered list of provider names to try.  Defaults to
        ``settings.provider_priority``.
    """

    def __init__(
        self,
        priority: Optional[List[str]] = None,
    ) -> None:
        self._priority: List[str] = priority or list(settings.provider_priority)
        self._providers: Dict[str, ProviderCallable] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, callable_: ProviderCallable) -> None:
        """
        Register a provider callable under *name*.

        Creates the corresponding circuit breaker automatically.

        Parameters
        ----------
        name:
            Provider identifier, e.g. ``"openai"``.
        callable_:
            Async function ``(request) → ChatCompletionResponse``.
        """
        self._providers[name] = callable_
        self._circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=settings.cb_failure_threshold,
            recovery_timeout=settings.cb_recovery_timeout,
            success_threshold=settings.cb_success_threshold,
        )
        logger.info("Router: registered provider '%s'", name)

    # ── Routing ───────────────────────────────────────────────────────────────

    async def route(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Route *request* to the best available provider.

        Algorithm
        ---------
        1. Build the candidate list:
             - If ``request.provider`` is set, use only that provider.
             - Otherwise, use the full priority list.
        2. Skip any provider whose circuit is OPEN.
        3. Call the provider; record success / failure.
        4. On success, return immediately.
        5. If all candidates fail, raise ``AllProvidersFailedError``.

        Returns
        -------
        ChatCompletionResponse
            The response from the first successful provider.

        Raises
        ------
        AllProvidersFailedError
            When no provider can serve the request.
        """
        candidates = self._build_candidate_list(request.provider)
        if not candidates:
            raise AllProvidersFailedError({})

        provider_errors: Dict[str, Exception] = {}

        for name in candidates:
            cb = self._circuit_breakers.get(name)
            if cb is None:
                logger.warning("Router: unknown provider '%s', skipping", name)
                continue

            if not cb.allow_request():
                logger.info(
                    "Router: circuit for '%s' is OPEN – skipping", name
                )
                provider_errors[name] = CircuitOpenError(name)
                continue

            try:
                start = time.perf_counter()
                logger.info("Router: trying provider '%s'", name)

                response = await self._providers[name](request)

                elapsed = time.perf_counter() - start
                cb.record_success()

                # Attach gateway metadata to the response
                response.gateway_meta.update(
                    {
                        "provider_used": name,
                        "latency_ms": round(elapsed * 1000, 2),
                        "providers_tried": list(provider_errors.keys()) + [name],
                    }
                )

                logger.info(
                    "Router: '%s' succeeded in %.1f ms", name, elapsed * 1000
                )
                return response

            except Exception as exc:
                logger.warning(
                    "Router: provider '%s' failed: %s: %s",
                    name,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                cb.record_failure()
                provider_errors[name] = exc
                # Continue to next provider

        raise AllProvidersFailedError(provider_errors)

    # ── Health reporting ──────────────────────────────────────────────────────

    def provider_statuses(self) -> List[ProviderStatus]:
        """
        Return the health and circuit-breaker status for every registered
        provider.  Used by the ``GET /health`` endpoint.
        """
        import datetime

        statuses: List[ProviderStatus] = []
        for name, cb in self._circuit_breakers.items():
            last_fail = cb.last_failure_at
            last_fail_iso: Optional[str] = None
            if last_fail is not None:
                # Convert monotonic timestamp to an approximate wall-clock ISO string
                # (monotonic - now) gives the delta; apply to current UTC time
                delta = time.monotonic() - last_fail
                approx_utc = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                    seconds=delta
                )
                last_fail_iso = approx_utc.isoformat()

            statuses.append(
                ProviderStatus(
                    name=ProviderName(name),
                    healthy=cb.state == CircuitState.CLOSED,
                    circuit_state=cb.state.value,
                    failure_count=cb.failure_count,
                    last_failure_at=last_fail_iso,
                )
            )
        return statuses

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_candidate_list(
        self, pinned: Optional[ProviderName]
    ) -> List[str]:
        """
        Build the ordered list of provider names to try for this request.

        - If *pinned* is set, returns a single-element list (that provider).
        - Otherwise, returns the priority list filtered to registered providers.
        """
        if pinned is not None:
            return [pinned.value]
        return [p for p in self._priority if p in self._providers]
