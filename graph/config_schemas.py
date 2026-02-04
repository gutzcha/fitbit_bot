from pydantic import BaseModel, Field
from typing import Optional

class LLMConfig(BaseModel):
    """Standard configuration for any LLM instance."""
    model: str
    temperature: float = 0.0
    streaming: bool = False
    max_tokens: Optional[int] = None