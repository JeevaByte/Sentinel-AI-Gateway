"""
main.py – Sentinel AI Gateway entry point.

Endpoints
---------
POST /chat                  Accept {prompt, user_id}, return ChatResponse.
GET  /health                Overall gateway health + per-provider status.
GET  /providers/status      Circuit-breaker state per provider.
GET  /metrics               Prometheus-compatible text exposition.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProviderStatus,
)
from reliability.router import ProviderRouter, get_provider_statuses

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total number of /chat requests",
    ["provider", "status"],
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_ms",
    "End-to-end latency of /chat requests in milliseconds",
    ["provider"],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
)
FALLBACK_COUNT = Counter(
    "gateway_fallback_total",
    "Number of times a fallback provider was used",
)
PROVIDER_CIRCUIT_STATE = Gauge(
    "gateway_provider_circuit_open",
    "1 if circuit breaker is open for the provider, 0 otherwise",
    ["provider"],
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentinel AI Gateway",
    description="Resilient multi-LLM gateway with circuit-breaker fallback",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_router = ProviderRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_health_status() -> HealthResponse:
    statuses = get_provider_statuses()
    overall = "healthy"
    open_count = sum(1 for s in statuses if s.state == "open")
    if open_count == len(statuses):
        overall = "degraded"
    elif open_count > 0:
        overall = "partial"
    return HealthResponse(status=overall, providers=statuses)


def _update_circuit_gauges() -> None:
    for ps in get_provider_statuses():
        PROVIDER_CIRCUIT_STATE.labels(provider=ps.name).set(
            1 if ps.state == "open" else 0
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a prompt to the best available LLM provider with automatic fallback."""
    try:
        response = await _router.chat(prompt=request.prompt, user_id=request.user_id)
    except RuntimeError as exc:
        REQUEST_COUNT.labels(provider="none", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    REQUEST_COUNT.labels(provider=response.provider_used, status="success").inc()
    REQUEST_LATENCY.labels(provider=response.provider_used).observe(response.latency_ms)
    if response.fallback_triggered:
        FALLBACK_COUNT.inc()
    _update_circuit_gauges()
    return response


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health() -> HealthResponse:
    """Return overall gateway health and per-provider circuit-breaker state."""
    return _build_health_status()


@app.get(
    "/providers/status",
    response_model=list[ProviderStatus],
    status_code=status.HTTP_200_OK,
)
async def providers_status() -> list[ProviderStatus]:
    """Return circuit-breaker state for every configured provider."""
    return get_provider_statuses()


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    _update_circuit_gauges()
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
