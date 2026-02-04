"""
graph/schemas.py

Domain models and schema definitions for the Fitbit Assistant.
This file contains the "Data Layer" of the application.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import (BaseModel, ConfigDict, Field, confloat, field_validator,
                      model_validator)

from graph.data_config import (INTENT_MIN_SOURCES, INTENT_RESPONSE_TYPE,
                               SOURCE_ORDER)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROUTING TYPES (High-Level Strategy)
# ─────────────────────────────────────────────────────────────────────────────

DataSource = Literal[
    "USER_METRICS",  # Query time-series databases (e.g., SQL tables for Steps, HR, Sleep).
    "USER_PROFILE",  # Look up static user attributes (e.g., Name, Age, Goals).
    "KNOWLEDGE_BASE",  # Retrieve medical/health context (e.g., "Normal HR for age 30").
    "CALENDAR",  # Check user's schedule (e.g., "Do I have time for a walk?").
    "NONE",  # No data lookup required (e.g., Greetings, Out of scope).
]

IntentLabel = Literal[
    "METRIC_RETRIEVAL",  # User wants to see data.
    "CORRELATION_ANALYSIS",  # User wants to understand connections.
    "COACHING_REQUEST",  # User wants advice.
    "BENCHMARK_EVALUATION",  # User wants comparisons.
    "DATA_AVAILABILITY",  # System questions.
    "OUT_OF_SCOPE",  # Unrelated queries.
    "GREETING",  # Social pleasantries.
    "UNCLEAR",  # Maybe related but intent is unclear.
]

ResponseType = Literal[
    "DATA_LOOKUP",  # Direct answer with numbers.
    "TREND_ANALYSIS",  # analytical narrative.
    "ACTIONABLE_ADVICE",  # specific, grounded suggestion.
    "BENCHMARK_INFO",  # context-aware comparison.
    "HELP_MESSAGE",  # Canned response.
    "CLARIFICATION",  # System needs to ask a question.
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROVENANCE TYPES (Low-Level Execution)
# ─────────────────────────────────────────────────────────────────────────────

SourceKind = Literal[
    "sql",  # Value came from a specific database query.
    "user_profile",  # Value came from the loaded JSON profile object.
    "curated_kb",  # Value came from a specific medical guideline document.
    "conversation",  # Value came from a previous turn in chat history.
    "computed",  # Value was calculated on the fly.
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROVENANCE MODELS
# ─────────────────────────────────────────────────────────────────────────────


class SourceRef(BaseModel):
    """Provenance record for a single piece of information."""

    kind: SourceKind = Field(description="The technical origin of the data.")
    sql_table: Optional[str] = None
    sql_query: Optional[str] = None
    profile_path: Optional[str] = None
    profile_field: Optional[str] = None
    kb_name: Optional[str] = None
    kb_entry_id: Optional[str] = None
    kb_title: Optional[str] = None
    kb_content: Optional[str] = None
    kb_category: Optional[str] = None
    computed_from: Optional[List[str]] = None
    computation: Optional[str] = None
    fetched_at_iso: str = Field(description="ISO timestamp.")


class Fact(BaseModel):
    """A single atomic fact used by the assistant."""

    key: str = Field(description="Unique snake_case identifier.")
    value: Any = Field(description="The actual value.")
    unit: Optional[str] = None
    refs: List[SourceRef] = Field(description="List of SourceRefs.")

    @field_validator("refs")
    @classmethod
    def at_least_one_ref(cls, v: List[SourceRef]) -> List[SourceRef]:
        if not v:
            raise ValueError("Fact must have at least one SourceRef.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIT TRAIL MODELS
# ─────────────────────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CuratedKBEntry(BaseModel):
    entry_id: str
    kb_name: str
    title: str
    category: str
    content: str
    relevance_score: Optional[float] = None
    tags: List[str] = []


class CuratedKBQuery(BaseModel):
    kb_name: str
    query: str
    entries: List[CuratedKBEntry]


# ─────────────────────────────────────────────────────────────────────────────
# 5. USER PROFILE MODELS
# ─────────────────────────────────────────────────────────────────────────────


class Demographics(BaseModel):
    age_years: int
    sex: Optional[str] = None
    height_cm: Optional[float] = None


class BodyMetrics(BaseModel):
    weight_kg: Optional[float] = None
    weight_lbs: Optional[float] = None
    bmi: Optional[float] = None
    body_fat_pct: Optional[float] = None
    weight_last_updated_iso: Optional[str] = None


class Baselines(BaseModel):
    baseline_window_days: int
    avg_steps_per_day: Optional[float] = None
    avg_calories_per_day: Optional[float] = None
    avg_sleep_minutes_per_night: Optional[float] = None
    avg_time_in_bed_minutes_per_night: Optional[float] = None
    avg_resting_hr_bpm: Optional[float] = None
    avg_hr_bpm: Optional[float] = None
    avg_very_active_minutes_per_day: Optional[float] = None
    avg_sedentary_minutes_per_day: Optional[float] = None
    hrv_ms: Optional[float] = None


class ActivityProfile(BaseModel):
    activity_level: Optional[str] = None
    preferred_workout_types: List[str] = []
    timezone: Optional[str] = None


class HealthGoals(BaseModel):
    daily_steps_goal: Optional[int] = None
    weekly_active_minutes_goal: Optional[int] = None
    sleep_hours_goal: Optional[float] = None
    weight_goal_kg: Optional[float] = None


class CoachingPreferences(BaseModel):
    suggestiveness: float = 0.5
    tone: Optional[str] = None
    notification_frequency: Optional[str] = None


class SystemState(BaseModel):
    onboarding_completed: bool = False
    consent_medical_disclaimer: bool = False
    last_interaction_iso: Optional[str] = None
    last_suggestion_key: Optional[str] = None


class UserProfile(BaseModel):
    user_id: int
    user_name: str
    demographics: Demographics
    body_metrics: BodyMetrics
    baselines: Baselines
    activity_profile: ActivityProfile
    health_goals: HealthGoals
    coaching_preferences: CoachingPreferences
    system_state: SystemState
    model_config = ConfigDict(validate_assignment=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONVERSATION CONTEXT
# ─────────────────────────────────────────────────────────────────────────────


class ConversationState(BaseModel):
    """Lightweight conversation context."""

    current_topic: Optional[str] = None
    mentioned_metrics: set = Field(default_factory=set)
    turn_count: int = 0
    prior_intent: Optional[str] = None
    user_explicitly_asked: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTENT ROUTING METADATA
# ─────────────────────────────────────────────────────────────────────────────
LOW_CONF_OVERRIDE = 0.6


class IntentMetadata(BaseModel):
    """
    Strict routing signals.
    """

    intent: IntentLabel = Field(description="Router-safe intent label.")
    confidence: confloat(ge=0.0, le=1.0) = Field(
        description="Confidence score in [0,1]."
    )
    suggested_sources: List[DataSource] = Field(default_factory=list)
    response_type: ResponseType = Field(description="Type of response required.")

    mentioned_metrics: List[str] = Field(
        default_factory=list, description="Specific metrics like 'steps', 'hr'."
    )
    current_topic: Optional[str] = Field(
        default="general", description="High-level topic like 'activity' or 'sleep'."
    )

    is_followup: bool = Field(default=False)
    needs_clarification: bool = Field(default=False)

    @model_validator(mode="after")
    def enforce_policy(self) -> "IntentMetadata":
        # 1. Enforce minimum sources
        required = INTENT_MIN_SOURCES.get(self.intent, set())
        got = list(self.suggested_sources or [])
        got_set = set(got)

        missing = [s for s in required if s not in got_set]
        if missing:
            got.extend(missing)

        # 2. Strict clean-up for non-data intents
        if self.intent in ("GREETING", "OUT_OF_SCOPE"):
            got = ["NONE"]

        # 3. Sort and Deduplicate
        got_set = set(got)
        self.suggested_sources = [s for s in SOURCE_ORDER if s in got_set]

        # 4. Enforce Response Type Consistency
        expected = INTENT_RESPONSE_TYPE.get(self.intent, "DATA_LOOKUP")

        if not self.needs_clarification:
            if self.response_type == "CLARIFICATION":
                self.needs_clarification = True
            else:
                self.response_type = expected

        # 5. Low Confidence Override
        if self.confidence < LOW_CONF_OVERRIDE:
            self.needs_clarification = True

        return self
