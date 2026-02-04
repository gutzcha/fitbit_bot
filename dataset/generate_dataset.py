from dataset.fitbit_data_ingestion_sqlite3 import FitbitDataPipeline
from populate_user_profile import generate_all_user_profiles
from health_kb_loader import generate_health_knowledge_base
from inspect_db import inspect_database

# Generate SQL dataset
pipeline = FitbitDataPipeline()
pipeline.run_full_pipeline()

# Generate user profiles based on SQL dataset
generate_all_user_profiles()

# Generate knowledge base dataset
generate_health_knowledge_base()
# Inspect knowledge base
inspect_database()