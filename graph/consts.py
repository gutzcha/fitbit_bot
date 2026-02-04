import os
from pathlib import Path

CURRENT_DATE = "2016-04-11" # corresponding to the last entry in the dataset

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# ENV
ENV_PATH = BASE_DIR / ".env"

# Database
# --- Kaggle Config ---
KAGGLE_DATASET_ID = "arashnic/fitbit"

# --- File System Paths ---
# Use absolute paths or relative to the project root
PROCESSED_DIR = DATASET_DIR / "clean"
DB_NAME = "fitbit.sqlite"
DB_PATH = DATASET_DIR / "db" / DB_NAME

# Known subdirectories inside the downloaded zip, there are two folders but we only use one here.
RAW_DATA_SUBDIRS = [
    "mturkfitbit_export_3.12.16-4.11.16",
    "Fitabase Data 3.12.16-4.11.16"
]


DB_URI = f"sqlite:///{DB_PATH}"

# Knowledge base
KB_NAME = "health_kb"
EMBED_PROVIDER = "ollama" # "openai
EMBED_MODEL = "mxbai-embed-large:335m"
KB_RAW_DATA_PATH = DATASET_DIR / "health_data"
KB_PATH =DATASET_DIR / "db" / "health_kb"


# Profiles
MOCK_USER_CONFIGS = {
    1503960366: {
        "demographics": {"age": 32, "sex": "male", "height_cm": 178},
        "timezone": "Asia/Jerusalem",
        "health_goals": {"daily_steps_goal": 12000, "weight_goal_kg": 75.0},
        "preferred_workout_types": ["running", "swimming"],
        "coaching_preferences": {"suggestiveness": "high", "tone": "energetic"}
    },
    2022484408: {
        "demographics": {"age": 55, "sex": "male", "height_cm": 170},
        "timezone": "Asia/Jerusalem",
        "health_goals": {"daily_steps_goal": 12000, "weight_goal_kg": 80.0},
        "preferred_workout_types": ["walking"],
        "coaching_preferences": {"suggestiveness": "high", "tone": "energetic"}
    },
    # Add other users here or use a default fallback
}
DEFAULT_CONFIG = {
        "demographics": {},
        "timezone": "UTC",
        "health_goals": {"daily_steps_goal": 10000},
        "preferred_workout_types": ["walking"],
        "coaching_preferences": {"suggestiveness": "medium", "tone": "supportive"}
    }

PROFILE_DIR = os.path.join(DATASET_DIR, "user_profiles")
CHAT_CONFIG_PATH = os.path.join(BASE_DIR, "app", "config.json")


