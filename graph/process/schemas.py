# graph/process/schemas.py
"""
PROCESS SCHEMAS
===============

Defines the structured contract between interpretation (LLM planner)
and execution (fetchers, derivations, and response synthesis).

These schemas guarantee that:
- The planner can only refer to real metrics that exist in the database.
- Time ranges are explicit and machine-executable.
- Routing decisions (SQL vs KB vs clarification) are deterministic.
- The PROCESS subgraph can be audited, tested, and evaluated mechanically.
"""

from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, confloat, field_validator

from graph.schemas import DataSource, ResponseType

MetricName = Literal[
    "steps_daily",
    "distance_daily",
    "calories_daily",
    "active_minutes_daily",
    "activity_distance_breakdown_daily",
    "steps_hourly",
    "heart_rate",
    "weight_kg",
    "weight_lbs",
    "bmi",
    "body_fat_percent",
]

Aggregation = Literal["raw", "sum", "avg", "min", "max"]
Granularity = Literal["hour", "day", "week", "month"]


class TimeRange(BaseModel):
    start_date: Optional[date] = Field(default=None)
    end_date: Optional[date] = Field(default=None)
    granularity: Optional[Granularity] = Field(default="day")


class MetricSpec(BaseModel):
    name: MetricName
    aggregation: Aggregation = Field(default="raw")
    description: Optional[str] = None


class PlanStep(BaseModel):
    step_id: str
    action: Literal[
        "fetch_user_metrics_sql",
        "fetch_user_profile",
        "fetch_knowledge_base",
        "compute_derived_stats",
        "synthesize_answer",
    ]
    notes: Optional[str] = None


class ProcessPlan(BaseModel):
    needs_clarification: bool = Field(default=False)
    clarification_question: Optional[str] = None

    response_type: ResponseType = Field(default="DATA_LOOKUP")
    selected_sources: List[DataSource] = Field(default_factory=list)

    metrics: List[MetricSpec] = Field(default_factory=list)
    time_range: Optional[TimeRange] = None

    steps: List[PlanStep] = Field(default_factory=list)

    confidence: confloat(ge=0.0, le=1.0) = Field(default=0.7)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v):
        if v is None:
            return 0.7
        try:
            fv = float(v)
        except Exception:
            return 0.7

        # Accept percentage style outputs
        if fv > 1.0 and fv <= 100.0:
            return fv / 100.0
        # Clamp weird outputs
        if fv < 0.0:
            return 0.0
        if fv > 1.0:
            return 1.0
        return fv

    def route_tags(self) -> List[str]:
        tags: List[str] = []
        if self.needs_clarification:
            tags.append("needs_clarification")
        if "USER_METRICS" in self.selected_sources:
            tags.append("uses_sql")
        if "KNOWLEDGE_BASE" in self.selected_sources:
            tags.append("uses_kb")
        return tags


class ExecutionResponse(BaseModel):
    """
    Structured output for the Execution Agent.
    """

    answer: str = Field(
        description="The final comprehensive answer to the user, synthesizing data from all tools."
    )
    confidence: float = Field(
        description="A score between 0.0 and 1.0 indicating confidence in the answer based on available data."
    )
    needs_clarification: bool = Field(
        default=False,
        description="Set to True ONLY if critical data is missing and you cannot answer safely.",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="If needs_clarification is True, provide the specific question to ask the user.",
    )


class GroundingMetadata(BaseModel):
    """
    Captures the proof that the agent actually retrieved data.
    Stored in the AssistantState for auditing.
    """

    sql_queries: List[str] = Field(
        default_factory=list, description="Actual SQL executed."
    )
    table_names: List[str] = Field(default_factory=list, description="Tables accessed.")
    confidence: float = Field(default=1.0, description="Confidence score (0.0 to 1.0).")


class SQLAgentResponse(GroundingMetadata):
    """
    The structured response expected from the Data Agent.
    Inherits fields (sql_queries, table_names, confidence) from GroundingMetadata.
    Adds the 'answer' field for the natural language response.
    """

    answer: str = Field(
        description="The natural language answer to the user's question."
    )
