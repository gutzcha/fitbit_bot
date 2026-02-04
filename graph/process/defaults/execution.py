"""
graph/defaults/execution.py
"""

from pydantic import BaseModel, Field
from graph.config_schemas import LLMConfig

class ExecutionNodeConfig(BaseModel):
    """
    Configuration for the Execution Agent (The Manager).
    """
    llm: LLMConfig
    max_iterations: int = Field(default=5, ge=1, description="Max ReAct loops before stopping.")