from typing import Optional

from pydantic import BaseModel, Field

from graph.config_schemas import LLMConfig


class IntentNodeConfig(BaseModel):
    """Specific configuration for the Intent Node."""

    llm_fast: LLMConfig
    llm_slow: Optional[LLMConfig] = None

    # Defaults are defined here, "close" to the data definition
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_history_limit: int = Field(default=10, gt=0)
