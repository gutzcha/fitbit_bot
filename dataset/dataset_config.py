"""
dataset.dataset_config.py
# This dictionary defines how to process every table.
# To add a new table, you only need to edit this dictionary.
"""
from typing import Dict, Optional, TypedDict, Literal
from graph.consts import (EMBED_MODEL, KB_PATH, KB_RAW_DATA_PATH, KB_NAME, EMBED_PROVIDER)

class TableConfig(TypedDict):
    csv_name: str
    rename_map: Dict[str, str]
    date_cols: Dict[str, Optional[str]]

SCHEMA_MAPPING: Dict[str, TableConfig] = {
    "daily_activity": {
        "csv_name": "dailyActivity_merged.csv",
        "rename_map": {
            "Id": "user_id",
            "ActivityDate": "event_date",
            "TotalSteps": "total_steps",
            "TotalDistance": "total_distance",
            "VeryActiveMinutes": "very_active_minutes",
            "FairlyActiveMinutes": "fairly_active_minutes",
            "LightlyActiveMinutes": "lightly_active_minutes",
            "SedentaryMinutes": "sedentary_minutes",
            "Calories": "calories",
        },
        "date_cols": {"event_date": "%m/%d/%Y"},
    },
    "sleep_day": {
        "csv_name": "sleepDay_merged.csv",
        "rename_map": {
            "Id": "user_id",
            "SleepDay": "event_date",
            "TotalSleepRecords": "sleep_records",
            "TotalMinutesAsleep": "minutes_asleep",
            "TotalTimeInBed": "time_in_bed",
        },
        "date_cols": {"event_date": None}, # Auto-detect format
    },
    "heartrate": {
        "csv_name": "heartrate_seconds_merged.csv",
        "rename_map": {
            "Id": "user_id",
            "Time": "event_time",
            "Value": "bpm"
        },
        "date_cols": {"event_time": None},
    },
    "hourly_steps": {
        "csv_name": "hourlySteps_merged.csv",
        "rename_map": {
            "Id": "user_id",
            "ActivityHour": "event_time",
            "StepTotal": "steps",
        },
        "date_cols": {"event_time": None},
    },
    "weight_log": {
        "csv_name": "weightLogInfo_merged.csv",
        "rename_map": {
            "Id": "user_id",
            "Date": "event_time",
            "WeightKg": "weight_kg",
            "WeightPounds": "weight_lbs",
            "BMI": "bmi",
        },
        "date_cols": {"event_time": None},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CENTRALIZED CONFIGURATION for Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_PROVIDERS = Literal["ollama", "openai"]
INGESTION_CONFIG = {
    "paths": {
        "data_dir": KB_RAW_DATA_PATH,
        "db_dir": KB_PATH,
    },
    "models": {
        "embedding_model": EMBED_MODEL,
        "embedding_provider": EMBED_PROVIDER,
    },
    "vector_store": {
        "collection_name": KB_NAME,
        "distance_function": "cosine",
    },
    "processing": {
        "batch_size": 100,  # Prevents local OOM or timeout
        "max_retries": 3,
    },
    "default_chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },
}

STRATEGY_MAP = {
    "Blood_Oxygen.txt": {"size": 400, "overlap": 0, "cat": "Vital Signs"},
    "Blood_Pressure.txt": {"size": 500, "overlap": 0, "cat": "Vital Signs"},
    "Vital_Signs.txt": {"size": 500, "overlap": 0, "cat": "Vital Signs"},
    "Heart_Rate_Zones.txt": {"size": 500, "overlap": 0, "cat": "Heart Health"},
    "VO2_Max.txt": {"size": 500, "overlap": 0, "cat": "Fitness"},
    "Questions_and_Answers.txt": {"size": 600, "overlap": 0, "cat": "Q&A"},
    "Exercise_and_Fitness.txt": {"size": 1000, "overlap": 150, "cat": "Fitness"},
    "Nutrition.txt": {"size": 1000, "overlap": 150, "cat": "Nutrition"},
    "General_Health.txt": {"size": 1100, "overlap": 200, "cat": "General"},
    "Mental_Health.txt": {"size": 700, "overlap": 100, "cat": "Mental Health"},
    "Chronic_Conditions.txt": {
        "size": 800,
        "overlap": 100,
        "cat": "Chronic Conditions",
    },
    "Workout_Plans_and_Goals.txt": {"size": 1200, "overlap": 200, "cat": "Fitness"},
}


from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# SQL AVAILABLE METRICS
# derived from graph.dataset_config.SCHEMA_MAPPING
# ─────────────────────────────────────────────────────────────────────────────
from typing import Dict, List

# -----------------------------------------------------------------------------
# SQL AVAILABLE METRICS
# Derived from the inspected SQLite schema (inspect_db.py output)
# -----------------------------------------------------------------------------

SQL_AVAILABLE_METRICS: Dict[str, List[str]] = {
    "daily_activity": [
        "total_steps",
        "total_distance",
        "trackerdistance",
        "loggedactivitiesdistance",
        "veryactivedistance",
        "moderatelyactivedistance",
        "lightactivedistance",
        "sedentaryactivedistance",
        "very_active_minutes",
        "fairly_active_minutes",
        "lightly_active_minutes",
        "sedentary_minutes",
        "calories",
    ],
    "heartrate": [
        "bpm",
    ],
    "hourly_steps": [
        "steps",
    ],
    "weight_log": [
        "weight_kg",
        "weight_lbs",
        "fat",
        "bmi",
        "ismanualreport",
        "logid",
    ],
}

SQL_SCHEMA = """
/* FITBIT DATABASE SCHEMA
  Database: SQLite
  Common Keys: user_id (INTEGER), event_date/event_time (TIMESTAMP)
  Generated from local inspection output.
*/

-- 1. Daily Activity Summary
CREATE TABLE daily_activity (
    user_id                   INTEGER,
    event_date                TIMESTAMP,

    total_steps               INTEGER,
    total_distance            REAL,

    trackerdistance           REAL,
    loggedactivitiesdistance  REAL,
    veryactivedistance        REAL,
    moderatelyactivedistance  REAL,
    lightactivedistance       REAL,
    sedentaryactivedistance   REAL,

    very_active_minutes       INTEGER,
    fairly_active_minutes     INTEGER,
    lightly_active_minutes    INTEGER,
    sedentary_minutes         INTEGER,

    calories                  INTEGER
);

-- 2. Heart Rate (High Frequency)
CREATE TABLE heartrate (
    user_id     INTEGER,
    event_time  TIMESTAMP,
    bpm         INTEGER
);

-- 3. Hourly Steps
CREATE TABLE hourly_steps (
    user_id     INTEGER,
    event_time  TIMESTAMP,
    steps       INTEGER
);

-- 4. Weight Logs
CREATE TABLE weight_log (
    user_id          INTEGER,
    event_time       TIMESTAMP,
    weight_kg        REAL,
    weight_lbs       REAL,
    fat              REAL,
    bmi              REAL,
    ismanualreport   INTEGER,
    logid            INTEGER
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR KNOWLEDGE TOPICS
# derived from index.txt content descriptions
# ─────────────────────────────────────────────────────────────────────────────
VECTOR_KNOWLEDGE_TOPICS: Dict[str, List[str]] = {
    "Vital Signs & Biometrics": [
        "Blood Oxygen (SpO2, Hypoxia, Sleep fluctuations)",
        "Blood Pressure (Hypertension stages, Systolic/Diastolic definitions)",
        "Body Fat Percentage (Ranges by age/gender, Measurement methods)",
        "Vital Signs (Resting HR, Respiration rate, Body temperature)",
    ],
    "Heart Health": [
        "Heart Rate Zones (Training zones, Max HR formulas)",
        "VO2 Max (Aerobic capacity, Improvement strategies)",
    ],
    "Fitness & Workouts": [
        "Exercise & Fitness (HIIT, Strength training, Overtraining, Warm-up/Cool-down)",
        "Workout Plans (5K/10K/Marathon training, Muscle gain, Weight loss routines)",
    ],
    "Nutrition & Diet": [
        "Nutrition Basics (Macros, Micronutrients, Hydration, Fiber)",
        "Dietary Strategy (Meal planning, Reading labels, Sugar/Sodium reduction)",
    ],
    "Mental Health": [
        "Mental Wellbeing (Stress management, Anxiety, Depression, Mindfulness)",
    ],
    "Specific Conditions & Demographics": [
        "Chronic Conditions (Diabetes, Hypertension, Long COVID, Kidney stones)",
        "Men's Health (Testosterone, Prostate health)",
        "Women's Health (Menopause, PCOS, Pregnancy, Postpartum)",
        "Senior's Health (Fall prevention, Dementia signs, Mobility)",
        "Reproductive Health (Contraception, HPV, Libido)",
    ],
    "General Health": [
        "General Preventive Care (Immune health, Sleep hygiene, Posture)",
        "Q&A (Common health questions and structured answers)",
    ],
}