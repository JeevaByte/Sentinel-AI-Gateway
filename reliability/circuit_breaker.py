"""
reliability/circuit_breaker.py
───────────────────────────────
Per-provider circuit breaker that prevents the gateway from hammering a
provider that is already failing.

State machine
─────────────
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  CLOSED ──(n failures in a row)──► OPEN ──(timeout)──► HALF_OPEN│
   │    ▲                                                      │      │
   │    └─────────────(m successes in a row)────────────────── ┘      │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘

Usage::

    cb = CircuitBreaker(name="openai")

    if cb.allow_request():
        try:
            result = await call_provider(...)
            cb.record_success()
        except Exception as exc:
            cb.record_failure()
            raise
    else:
        raise CircuitOpenError("openai circuit is OPEN")
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State enum
# ─────────────────────────────────────────────────────────────────────────────


class CircuitState(str, Enum):
    """Possible circuit-breaker states."""

    CLOSED = "closed"       # Normal operation – requests pass through
    OPEN = "open"           # Failing – requests are rejected immediately
    HALF_OPEN = "half_open" # Recovery probe – limited requests allowed


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the circuit is OPEN."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Circuit breaker for provider '{provider}' is OPEN. "
            "Request rejected to protect system stability."
        )
        self.provider = provider


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────────


class CircuitBreaker:
    """
    Thread-safe circuit breaker for a single LLM provider.

    Parameters
    ----------
    name:
        Human-readable provider name used in logs and error messages.
    failure_threshold:
        Number of consecutive failures that trip the circuit to OPEN.
    recovery_timeout:
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    success_threshold:
        Number of consecutive successes in HALF_OPEN state needed to
        close the circuit again.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        # Mutable state (protected by _lock)
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_at: Optional[float] = None  # epoch seconds
        self._lock = threading.Lock()

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may trigger OPEN → HALF_OPEN transition)."""
        with self._lock:
            return self._get_state_locked()

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def last_failure_at(self) -> Optional[float]:
        with self._lock:
            return self._last_failure_at

    def allow_request(self) -> bool:
        """
        Return ``True`` if the circuit allows the request to proceed.

        Side-effect: may transition OPEN → HALF_OPEN when the recovery
        timeout has elapsed.
        """
        with self._lock:
            state = self._get_state_locked()
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                # Allow a single probe request through
                return True
            # OPEN state – reject
            return False

    def record_success(self) -> None:
        """
        Record a successful provider call.

        In HALF_OPEN state, accumulate successes toward closing the circuit.
        In CLOSED state, reset the failure counter.
        """
        with self._lock:
            state = self._get_state_locked()
            if state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif state == CircuitState.CLOSED:
                # Reset consecutive failure counter on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed provider call.

        Increments the failure counter.  In CLOSED state, trips the circuit
        to OPEN once the threshold is reached.  In HALF_OPEN state, reverts
        immediately back to OPEN.
        """
        with self._lock:
            state = self._get_state_locked()
            self._last_failure_at = time.monotonic()
            if state == CircuitState.HALF_OPEN:
                # Probe failed – go back to OPEN
                self._transition_to(CircuitState.OPEN)
            else:
                # CLOSED state – increment and check threshold
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """
        Manually reset the circuit breaker to CLOSED state.

        Useful in tests or after a manual operator intervention.
        """
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_state_locked(self) -> CircuitState:
        """
        Return the current state, handling the OPEN → HALF_OPEN time-based
        transition.  Must be called with ``_lock`` held.
        """
        if self._state == CircuitState.OPEN and self._last_failure_at is not None:
            elapsed = time.monotonic() - self._last_failure_at
            if elapsed >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """
        Perform a state transition and reset associated counters.
        Must be called with ``_lock`` held.
        """
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_at = None
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            self._last_failure_at = time.monotonic()
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        if old_state != new_state:
            logger.warning(
                "Circuit breaker '%s': %s → %s",
                self.name,
                old_state.value,
                new_state.value,
            )

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.value!r}, "
            f"failures={self._failure_count})"
        )
