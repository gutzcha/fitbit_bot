from pydantic import BaseModel, Field
from graph.config_schemas import LLMConfig


class ClarificationNodeConfig(BaseModel):
    """Configuration for the Clarification Node."""
    llm: LLMConfig
    max_history_limit: int = Field(default=10, gt=0)
