from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str
    user_id: str


class ChatResponse(BaseModel):
    response: str
    provider_used: str
    latency_ms: float
    fallback_triggered: bool
    attempt_count: int


class ProviderStatus(BaseModel):
    name: str
    state: str
    failure_count: int
    last_failure: Optional[datetime] = None


class HealthResponse(BaseModel):
    status: str
    providers: List[ProviderStatus]
