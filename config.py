import os
from dataclasses import dataclass
from typing import List


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int
    max_retries: int
    priority_order: int


def load_provider_configs() -> List[ProviderConfig]:
    providers = [
        ProviderConfig(
            name="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url="https://api.openai.com/v1",
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout_seconds=30,
            max_retries=3,
            priority_order=1,
        ),
        ProviderConfig(
            name="anthropic",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            base_url="https://api.anthropic.com/v1",
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            timeout_seconds=30,
            max_retries=3,
            priority_order=2,
        ),
        ProviderConfig(
            name="gemini",
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            timeout_seconds=30,
            max_retries=3,
            priority_order=3,
        ),
        ProviderConfig(
            name="crusoe",
            api_key=os.getenv("CRUSOE_API_KEY", ""),
            base_url="https://api.crusoe.ai/v1",
            model=os.getenv("CRUSOE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            timeout_seconds=30,
            max_retries=3,
            priority_order=4,
        ),
        ProviderConfig(
            name="ollama",
            api_key="",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            timeout_seconds=60,
            max_retries=2,
            priority_order=5,
        ),
    ]
    return sorted(providers, key=lambda p: p.priority_order)
