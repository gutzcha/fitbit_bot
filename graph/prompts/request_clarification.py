"""
graph/prompts/request_clarification.py

System prompt and finalized ChatPromptTemplate for the Clarification Node.
Dynamically injects available metrics/profile fields from config.
"""

from graph.data_config import (
    PROFILE_AVAILABLE_FIELDS,
    SQL_AVAILABLE_METRICS,
    VECTOR_KNOWLEDGE_TOPICS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────


def _format_metrics() -> str:
    return ", ".join(SQL_AVAILABLE_METRICS.keys())


def _format_profile() -> str:
    return ", ".join(PROFILE_AVAILABLE_FIELDS.keys())


def _format_topics() -> str:
    return ", ".join(VECTOR_KNOWLEDGE_TOPICS)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYSTEM STRING CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_SYSTEM_STRING = f"""
You are a helpful health data assistant.
The user's query is ambiguous or lacks specific details.
Your goal: Ask a SINGLE, concise question to clarify their intent.

AVAILABLE DATA SOURCES:
1. METRICS (I can graph/summarize these): {_format_metrics()}
2. PROFILE (I know these about the user): {_format_profile()}
3. KNOWLEDGE (I can answer general qs on): {_format_topics()}

RULES:
- Do NOT offer data we don't have (e.g. Blood Pressure, Stress, HRV).
- Specify all the metrics your can offer
- If they ask 'How am I doing?', list 2-3 key metrics (e.g. Sleep, Steps) as options.
- Keep it under 20 words.
"""