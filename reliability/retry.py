"""
reliability/retry.py
─────────────────────
Async retry decorator / helper built on top of **tenacity**.

Provides a pre-configured ``retry_async`` decorator that applies:
  • Exponential back-off with jitter
  • A configurable maximum number of attempts
  • Retry only on transient errors (network issues, 5xx, rate-limits)

Usage::

    from reliability.retry import with_retry

    @with_retry()
    async def call_provider(...):
        ...

    # OR call the helper directly:
    result = await retry_call(call_provider, arg1, kwarg=val)
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Coroutine, Optional, Tuple, Type

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Retryable exceptions
# ─────────────────────────────────────────────────────────────────────────────

#: Exception types that warrant a retry.  Extend this tuple as needed.
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    httpx.TimeoutException,       # Read / connect / write / pool timeout
    httpx.NetworkError,           # DNS failure, connection refused, …
    httpx.RemoteProtocolError,    # Unexpected response from server
)


def _is_retryable_status(exc: BaseException) -> bool:
    """
    Return True when an ``httpx.HTTPStatusError`` indicates a transient
    server-side error (429 rate-limit or 5xx).
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


def _should_retry(exc: BaseException) -> bool:
    """Predicate used by tenacity to decide whether to retry."""
    return isinstance(exc, RETRYABLE_EXCEPTIONS) or _is_retryable_status(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Logging callback
# ─────────────────────────────────────────────────────────────────────────────


def _log_retry(retry_state: RetryCallState) -> None:
    """Log each retry attempt for observability."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retry attempt %d/%d for '%s' due to %s: %s",
        retry_state.attempt_number,
        settings.retry_max_attempts,
        getattr(retry_state.fn, "__name__", "<unknown>"),
        type(exc).__name__ if exc else "unknown error",
        str(exc)[:120],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public decorator factory
# ─────────────────────────────────────────────────────────────────────────────


def with_retry(
    max_attempts: Optional[int] = None,
    initial_wait: Optional[float] = None,
    max_wait: Optional[float] = None,
) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:
    """
    Decorator factory that wraps an async function with retry logic.

    Parameters
    ----------
    max_attempts:
        Override ``settings.retry_max_attempts``.
    initial_wait:
        Override ``settings.retry_initial_wait`` (seconds, exponential base).
    max_wait:
        Override ``settings.retry_max_wait`` (seconds, exponential ceiling).

    Example
    -------
    ::

        @with_retry(max_attempts=5)
        async def call_openai(payload: dict) -> dict:
            ...
    """
    attempts = max_attempts or settings.retry_max_attempts
    initial = initial_wait or settings.retry_initial_wait
    maximum = max_wait or settings.retry_max_wait

    def decorator(
        func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        @wraps(func)
        @retry(
            # Only retry when _should_retry deems the exception transient
            retry=retry_if_exception(_should_retry),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=initial, max=maximum),
            before_sleep=_log_retry,
            reraise=True,
        )
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper

    return decorator
