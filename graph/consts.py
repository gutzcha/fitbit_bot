import os
from pathlib import Path

CURRENT_DATE = "2016-04-11"

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Database
DB_NAME = "fitbit.sqlite"  # Changed to sqlite as requested
DB_PATH = os.path.join(DATASET_DIR, "db", DB_NAME)
DB_URI = f"sqlite:///{DB_PATH}"

KB_PATH = Path("..") / "dataset" / "db" / "health_kb"

# Profiles
PROFILE_DIR = os.path.join(DATASET_DIR, "user_profiles")
CHAT_CONFIG_PATH = os.path.join(BASE_DIR, "app", "chat_config.json")

EMBED_MODEL = "mxbai-embed-large:335m"
ENV_PATH = Path("..")/ ".env"
