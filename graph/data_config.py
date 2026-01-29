"""
graph/data_config.py

Central configuration for Data Sources, Intent Definitions, and Knowledge Base topics.
"""

from typing import Dict, Set

# ─────────────────────────────────────────────────────────────────────────────
# 1. AVAILABLE DATA (The "Menu")
# ─────────────────────────────────────────────────────────────────────────────

SQL_AVAILABLE_METRICS = {
    "steps": "Daily step count and intensity",
    "heart_rate": "Time-series heart rate data (bpm)",
    "calories": "Daily calories burned",
    "active_minutes": "Very active, fairly active, and sedentary minutes",
    "weight": "Body weight logs (kg/lbs) and BMI",
}

PROFILE_AVAILABLE_FIELDS = {
    "age": "User age in years",
    "height": "User height in cm",
    "weight_goal": "Target weight",
    "step_goal": "Daily step target",
    "activity_level": "Sedentary, Active, etc.",
}

VECTOR_KNOWLEDGE_TOPICS = [
    "normal_heart_rate_ranges",
    "sleep_hygiene_tips",
    "zone_minutes_explanation",
    "bmi_categories",
    "step_goal_recommendations",
    "cardio_fitness_score_explained",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. INTENT & ROUTING CONFIGURATION (Moved from schemas.py)
# ─────────────────────────────────────────────────────────────────────────────

# Priority order for sources
SOURCE_ORDER = ["USER_METRICS", "USER_PROFILE", "KNOWLEDGE_BASE", "CALENDAR", "NONE"]

# Definitions used in System Prompts
INTENT_DEFINITIONS: Dict[str, str] = {
    "METRIC_RETRIEVAL": (
        "Descriptive request for specific metrics or simple trends. "
        "Examples: 'How many steps today?', 'Did I sleep well?'"
    ),
    "CORRELATION_ANALYSIS": (
        "Insight or correlative request seeking explanation of patterns. "
        "Examples: 'Why was my sleep worse?', 'Do my steps affect my resting heart rate?'"
    ),
    "COACHING_REQUEST": (
        "Prescriptive request for actionable advice grounded in user context. This can also include general health related questions"
        "Examples: 'How can I improve my sleep?', 'What should I do to hit my step goal?'"
    ),
    "BENCHMARK_EVALUATION": (
        "Benchmarking request comparing metrics to normal ranges. "
        "Examples: 'Is my heart rate normal?', 'Am I sleeping enough?'"
    ),
    "DATA_AVAILABILITY": (
        "Questions about what data exists or why something is missing. "
        "Examples: 'Do you have my heart rate data?', 'Why is there no sleep data?', 'What information can your provide'"
    ),
    "OUT_OF_SCOPE": (
        "Request unrelated to health, fitness, sleep, or physiological metrics. "
        "Examples: 'What's the weather?', 'Tell me a joke'"
    ),
    "GREETING": (
        "Greeting or capability check, route to a canned help message. "
        "Examples: 'Hello', 'Hi there', 'What can you do?'"
    ),
    "UNCLEAR": (
        "The user intent may be related to health but unclear and requires clarification"
        "Examples: 'How did i do?', 'How am I?', 'Did I asdf yesterday?'"
    ),
}

# Logic Rules for Validation
INTENT_MIN_SOURCES = {
    "METRIC_RETRIEVAL": {"USER_METRICS"},
    "CORRELATION_ANALYSIS": {"USER_METRICS", "USER_PROFILE"},
    "COACHING_REQUEST": {"USER_METRICS", "KNOWLEDGE_BASE"},
    "BENCHMARK_EVALUATION": {"USER_METRICS", "KNOWLEDGE_BASE"},
    "DATA_AVAILABILITY": {"USER_PROFILE"},
    "OUT_OF_SCOPE": {"NONE"},
    "GREETING": {"NONE"},
    "UNCLEAR": {"CLARIFICATION"},
}

INTENT_RESPONSE_TYPE = {
    "METRIC_RETRIEVAL": "DATA_LOOKUP",
    "CORRELATION_ANALYSIS": "TREND_ANALYSIS",
    "COACHING_REQUEST": "ACTIONABLE_ADVICE",
    "BENCHMARK_EVALUATION": "BENCHMARK_INFO",
    "DATA_AVAILABILITY": "DATA_LOOKUP",
    "OUT_OF_SCOPE": "HELP_MESSAGE",
    "GREETING": "HELP_MESSAGE",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. SQL SCHEMA CONTEXT (Pre-loaded for efficiency)
# ─────────────────────────────────────────────────────────────────────────────

DB_SCHEMA_CONTEXT = """
You have access to a SQLite database with the following tables. 
- The 'user_id' is always BIGINT.
- 'event_date' is a DATE string ('YYYY-MM-DD').
- 'event_time' is a TIMESTAMP string ('YYYY-MM-DD HH:MM:SS').
- This data is a snapshot representing activity during 2016-03-12 -> 2026-04-16

1. daily_activity
   - Columns: user_id, event_date, total_steps, total_distance, calories, very_active_minutes, fairly_active_minutes, lightly_active_minutes, sedentary_minutes.
   - Use for: Daily summaries. "How many steps did I take?", "Calories burned today".
   - IMPORTANT:
     * event_date is stored as a TIMESTAMP string like '2016-04-10 00:00:00' in this table.
     * Always compare using DATE(event_date) or a full timestamp
       Example:
       WHERE DATE(event_date) = '2016-04-10'
       or
       WHERE event_date = '2016-04-10 00:00:00'

2. heartrate
   - Columns: user_id, event_time, bpm.
   - Use for: Minute-level heart rate data.
   - WARNING: This table is massive. ALWAYS filter by a specific time range (e.g., WHERE event_time LIKE '2016-04-01%') and LIMIT results to avoid crashes.

3. hourly_steps
   - Columns: user_id, event_time, steps.
   - Use for: Intraday activity analysis. "When was I most active?", "Steps per hour".

4. weight_log
   - Columns: user_id, event_time, weight_kg, weight_lbs, bmi, fat.
   - Use for: Body metrics and weight tracking.
"""
