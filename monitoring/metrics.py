"""
monitoring/metrics.py
──────────────────────
Prometheus-based metrics collection for the Sentinel AI Gateway.

All metrics are registered once at module import time and updated by the
request lifecycle hooks defined at the bottom of this file.

Exposed metrics
───────────────
  gateway_requests_total          – Counter   (provider, model, status_code)
  gateway_request_latency_seconds – Histogram (provider, model)
  gateway_tokens_total            – Counter   (provider, model, token_type)
  gateway_circuit_breaker_state   – Gauge     (provider, state)
  gateway_retries_total           – Counter   (provider)

The metrics endpoint is mounted at ``/metrics`` by main.py using the
``make_asgi_app()`` helper from prometheus-client.

Usage from provider code::

    from monitoring.metrics import record_request, record_tokens

    record_request(provider="openai", model="gpt-4o", status_code=200, latency=0.42)
    record_tokens(provider="openai", model="gpt-4o", prompt=100, completion=50)
"""

from __future__ import annotations

import logging
from typing import Optional

from typing import Dict

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────────────────────────────────────


REQUEST_COUNTER = Counter(
    name="gateway_requests_total",
    documentation="Total number of chat-completion requests, by provider and outcome.",
    labelnames=["provider", "model", "status_code"],
)

LATENCY_HISTOGRAM = Histogram(
    name="gateway_request_latency_seconds",
    documentation="End-to-end request latency in seconds.",
    labelnames=["provider", "model"],
    # Buckets cover 50 ms up to 60 s – typical LLM latency range
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

TOKEN_COUNTER = Counter(
    name="gateway_tokens_total",
    documentation="Cumulative token usage, split by prompt / completion.",
    labelnames=["provider", "model", "token_type"],  # token_type: prompt | completion
)

CIRCUIT_BREAKER_STATE = Gauge(
    name="gateway_circuit_breaker_state",
    documentation=(
        "Current circuit-breaker state per provider. "
        "Values: 0=closed (healthy), 1=half_open, 2=open (unhealthy)."
    ),
    labelnames=["provider"],
)

RETRY_COUNTER = Counter(
    name="gateway_retries_total",
    documentation="Number of retry attempts, by provider.",
    labelnames=["provider"],
)

PROVIDER_ERROR_COUNTER = Counter(
    name="gateway_provider_errors_total",
    documentation="Provider-level errors, labelled by error type.",
    labelnames=["provider", "error_type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit-breaker state mapping
# ─────────────────────────────────────────────────────────────────────────────

_CB_STATE_VALUES: Dict[str, float] = {
    "closed": 0.0,
    "half_open": 1.0,
    "open": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions called by the request lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def record_request(
    *,
    provider: str,
    model: str,
    status_code: int,
    latency: float,
) -> None:
    """
    Record a completed request.

    Parameters
    ----------
    provider:
        The provider that handled the request (e.g. ``"openai"``).
    model:
        The model name reported in the response.
    status_code:
        HTTP-equivalent status code (200 for success, 500 for error, …).
    latency:
        Total elapsed time in **seconds** (not milliseconds).
    """
    REQUEST_COUNTER.labels(
        provider=provider,
        model=model,
        status_code=str(status_code),
    ).inc()

    LATENCY_HISTOGRAM.labels(
        provider=provider,
        model=model,
    ).observe(latency)


def record_tokens(
    *,
    provider: str,
    model: str,
    prompt: int,
    completion: int,
) -> None:
    """
    Record token usage for a completed request.

    Parameters
    ----------
    provider:
        Provider identifier.
    model:
        Model name.
    prompt:
        Number of prompt tokens consumed.
    completion:
        Number of completion tokens generated.
    """
    TOKEN_COUNTER.labels(
        provider=provider,
        model=model,
        token_type="prompt",
    ).inc(prompt)

    TOKEN_COUNTER.labels(
        provider=provider,
        model=model,
        token_type="completion",
    ).inc(completion)


def record_circuit_state(provider: str, state: str) -> None:
    """
    Update the Gauge that tracks circuit-breaker state.

    Parameters
    ----------
    provider:
        Provider identifier.
    state:
        One of ``"closed"``, ``"half_open"``, or ``"open"``.
    """
    numeric = _CB_STATE_VALUES.get(state, 0.0)
    CIRCUIT_BREAKER_STATE.labels(provider=provider).set(numeric)


def record_retry(provider: str) -> None:
    """Increment the retry counter for *provider*."""
    RETRY_COUNTER.labels(provider=provider).inc()


def record_provider_error(provider: str, error_type: str) -> None:
    """
    Increment the error counter for *provider*.

    Parameters
    ----------
    provider:
        Provider identifier.
    error_type:
        Short error class name, e.g. ``"TimeoutError"`` or ``"HTTPStatusError"``.
    """
    PROVIDER_ERROR_COUNTER.labels(
        provider=provider,
        error_type=error_type,
    ).inc()


# ─────────────────────────────────────────────────────────────────────────────
# ASGI sub-application
# ─────────────────────────────────────────────────────────────────────────────


def create_metrics_app() -> object:
    """
    Return an ASGI application that serves the Prometheus text exposition
    format at the path it is mounted on.

    Mount in main.py::

        from monitoring.metrics import create_metrics_app
        app.mount("/metrics", create_metrics_app())
    """
    return make_asgi_app()
