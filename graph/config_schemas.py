from typing import Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Standard configuration for any LLM instance."""

    model: str
    temperature: float = 0.0
    streaming: bool = False
    max_tokens: Optional[int] = None
