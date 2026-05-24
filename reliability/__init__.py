"""
reliability/__init__.py
────────────────────────
Package marker for the reliability layer.

Re-exports the most commonly used symbols so call-sites can write:

    from reliability import CircuitBreaker, Router, with_retry
"""

from reliability.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from reliability.retry import with_retry
from reliability.router import AllProvidersFailedError, Router

__all__ = [
    "AllProvidersFailedError",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "Router",
    "with_retry",
]
