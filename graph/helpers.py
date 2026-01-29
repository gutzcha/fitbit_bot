from dataclasses import dataclass
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from graph.consts import CURRENT_DATE

Provider = Literal["ollama", "anthropic"]
ModelType = Literal["slow", "fast"]


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    fast_model: str
    slow_model: str
    chat_model: BaseChatModel
    temperature: float = 0.0


OLLAM_CONFIG = ModelConfig(
    provider="ollama",
    fast_model="ministral-3:3b",
    slow_model="ministral-3:8b",
    temperature=0.0,
    chat_model=ChatOllama,
)

ANTHROPIC_CONFIG = ModelConfig(
    provider="ollama",
    fast_model="claude-haiku-4-5-20251001",
    slow_model="claude-sonnet-4-5-20250929",
    temperature=0.0,
    chat_model=ChatAnthropic,
)


def make_llm(provider: Provider, model_type: ModelType):
    """Factory to create fast and slow LLM instances."""
    if provider == "ollama":
        cfg = OLLAM_CONFIG
    elif provider == "anthropic":
        cfg = ANTHROPIC_CONFIG
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if model_type == "fast":
        llm = cfg.chat_model(model=cfg.fast_model, temperature=cfg.temperature)
    elif model_type == "slow":
        llm = cfg.chat_model(model=cfg.slow_model, temperature=cfg.temperature)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return llm


def get_fast_slow_llm(provider):
    fast_llm = make_llm(provider, model_type="fast")
    slow_llm = make_llm(provider, model_type="slow")
    return fast_llm, slow_llm


def get_current_date():

    # return datetime.now().strftime("%Y-%m-%d")
    # Debug POC: OVERRIDE CURRENT DATE since the data is not current.
    return CURRENT_DATE


import json
import os
from typing import Any, Dict


def load_config(config_file) -> Dict[str, Any]:
    """Load config from JSON or return defaults."""
    default_config = {
        "provider": "ollama",
        "slow_fallback_enabled": True,
        "slow_fallback_min_confidence": 0.8,
        "max_history_limit": 10,
    }
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except:
            return default_config
    return default_config
