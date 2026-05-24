"""
main.py
────────
Sentinel AI Gateway – FastAPI application entry point.

This module wires together every layer of the gateway:
  • Loads configuration (config.py)
  • Registers all LLM provider adapters with the Router
  • Mounts the Prometheus metrics ASGI sub-app at /metrics
  • Defines the HTTP endpoints:
      POST /v1/chat/completions  – main inference endpoint
      GET  /health               – provider health & circuit-breaker states
      GET  /                     – simple liveness ping
  • Instruments each request with latency / token / error metrics

Run locally::

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    GatewayError,
    GatewayHealthResponse,
    ProviderStatus,
)
from monitoring.metrics import (
    create_metrics_app,
    record_provider_error,
    record_request,
    record_tokens,
)
from providers import PROVIDER_MAP, PROVIDER_SHUTDOWN_HOOKS
from reliability.circuit_breaker import CircuitOpenError
from reliability.router import AllProvidersFailedError, Router

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application-level singletons
# ─────────────────────────────────────────────────────────────────────────────

#: Shared router instance (created during lifespan startup)
router: Router = Router()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup
    ───────
    1. Register every provider from ``PROVIDER_MAP`` with the router.
       The router creates a ``CircuitBreaker`` for each provider.

    Shutdown
    ────────
    1. Drain all provider HTTP clients to avoid resource leaks.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Sentinel AI Gateway starting up …")

    for name, fn in PROVIDER_MAP.items():
        router.register(name, fn)
        logger.info("Registered provider: %s", name)

    logger.info(
        "Provider priority: %s", " → ".join(settings.provider_priority)
    )
    logger.info("Gateway ready.")

    yield  # ← application serves requests here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Sentinel AI Gateway shutting down …")
    for close_fn in PROVIDER_SHUTDOWN_HOOKS:
        try:
            await close_fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing provider client: %s", exc)
    logger.info("All provider clients closed.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel AI Gateway",
    description=(
        "A resilient multi-LLM gateway with automatic failover, "
        "circuit breakers, and unified OpenAI-compatible API."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Restrict in production – update ``allow_origins`` to your front-end domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics sub-app ────────────────────────────────────────────────
app.mount("/metrics", create_metrics_app())


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log each incoming request and its response time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  %d  %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", tags=["Gateway"])
async def root() -> dict:
    """
    Liveness ping.

    Returns a simple JSON envelope confirming the gateway is running.
    Does not check provider health – use ``GET /health`` for that.
    """
    return {"status": "ok", "service": "Sentinel AI Gateway", "version": "0.1.0"}


@app.get(
    "/health",
    response_model=GatewayHealthResponse,
    tags=["Gateway"],
    summary="Provider health and circuit-breaker status",
)
async def health() -> GatewayHealthResponse:
    """
    Return the health state of each registered LLM provider.

    The ``status`` field summarises overall gateway health:
      - ``ok``       – all circuits closed (all providers healthy)
      - ``degraded`` – at least one circuit open but others still available
      - ``down``     – all circuits open (no providers available)
    """
    statuses = router.provider_statuses()
    healthy_count = sum(1 for s in statuses if s.healthy)

    if healthy_count == len(statuses):
        overall = "ok"
    elif healthy_count > 0:
        overall = "degraded"
    else:
        overall = "down"

    return GatewayHealthResponse(status=overall, providers=statuses)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Inference"],
    summary="Unified chat completions endpoint",
    status_code=status.HTTP_200_OK,
)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    Route a chat-completion request to the best available LLM provider.

    The gateway:
    1. Iterates the priority list, skipping any provider whose circuit is open.
    2. Calls the first available provider with automatic retry on transient errors.
    3. Records success / failure metrics and updates the circuit breaker.
    4. Returns a normalised ``ChatCompletionResponse`` regardless of which
       provider handled the request.

    If all providers fail, returns ``503 Service Unavailable``.
    """
    start = time.perf_counter()
    provider_used = "unknown"
    model_used = request.model or "unknown"

    try:
        response = await router.route(request)
        provider_used = response.provider.value
        model_used = response.model

        # Record success metrics
        elapsed = time.perf_counter() - start
        record_request(
            provider=provider_used,
            model=model_used,
            status_code=200,
            latency=elapsed,
        )
        record_tokens(
            provider=provider_used,
            model=model_used,
            prompt=response.usage.prompt_tokens,
            completion=response.usage.completion_tokens,
        )

        return response

    except AllProvidersFailedError as exc:
        elapsed = time.perf_counter() - start
        logger.error("All providers failed: %s", exc)

        # Record per-provider errors in metrics
        for name, err in exc.provider_errors.items():
            record_provider_error(provider=name, error_type=type(err).__name__)
            record_request(
                provider=name,
                model=model_used,
                status_code=503,
                latency=elapsed,
            )

        body = GatewayError(
            error="All LLM providers failed to process the request.",
            code="all_providers_failed",
            provider_errors=[
                {"provider": name, "error": str(err)}
                for name, err in exc.provider_errors.items()
            ],
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(),
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception("Unexpected gateway error: %s", exc)
        record_request(
            provider=provider_used,
            model=model_used,
            status_code=500,
            latency=elapsed,
        )
        body = GatewayError(
            error="An unexpected internal error occurred.",
            code="internal_error",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )
