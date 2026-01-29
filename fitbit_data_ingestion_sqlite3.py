"""
Fitbit Data Ingestion Pipeline (Standardized Schema Version)
Downloads Fitbit dataset, cleans/standardizes column names,
and saves to SQLite + CSV.
"""

import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import kagglehub
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FitbitDataPipeline:
    def __init__(self, dataset_dir: str = "dataset", db_name: str = "fitbit.sqlite"):
        self.dataset_dir = Path(dataset_dir)
        self.processed_dir = self.dataset_dir / "clean"
        self.db_path = self.dataset_dir / "db" / db_name
        self.csv_dir = None

    def download_dataset(self) -> None:
        """Step 1: Download Fitbit dataset from Kaggle."""
        logger.info("Step 1: Downloading dataset from Kaggle...")
        src = kagglehub.dataset_download("arashnic/fitbit")

        target_path = self.dataset_dir / "mturkfitbit_export_3.12.16-4.11.16"
        if not target_path.exists():
            try:
                shutil.copytree(src, self.dataset_dir, dirs_exist_ok=True)
                logger.info(f"Dataset copied to: {self.dataset_dir}")
            except Exception as e:
                logger.error(f"Failed to copy dataset: {e}")

    def initialize_storage(self) -> None:
        """Step 2: Create directories for DB and processed CSVs."""
        logger.info("Step 2: Initializing storage directories...")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Reset DB
        if self.db_path.exists():
            self.db_path.unlink()

        con = sqlite3.connect(self.db_path)
        con.close()

    def find_raw_csv_directory(self) -> Path:
        """Helper: Locate the directory containing the raw CSV files."""
        search_path = (
            self.dataset_dir
            / "mturkfitbit_export_3.12.16-4.11.16"
            / "Fitabase Data 3.12.16-4.11.16"
        )
        if search_path.exists():
            return search_path

        for root, _, files in os.walk(self.dataset_dir):
            if any(f.endswith(".csv") for f in files):
                return Path(root)
        raise FileNotFoundError(f"Could not find raw CSV files in {self.dataset_dir}")

    def _process_single_table(
        self,
        con: sqlite3.Connection,
        csv_name: str,
        table_name: str,
        rename_map: Dict[str, str],
        date_cols: Dict[str, str] = None,
    ):
        """
        Reads CSV, standardizes column names, fixes dates, and saves.

        Args:
            rename_map: Dict mapping raw CSV headers to standardized snake_case names.
            date_cols: Dict mapping {standardized_col_name: format_string or None}
        """
        file_path = self.csv_dir / csv_name
        if not file_path.exists():
            logger.warning(f"Skipping {table_name}: {csv_name} not found.")
            return

        logger.info(f"Processing {table_name}...")
        try:
            # 1. Read CSV
            df = pd.read_csv(file_path)

            # 2. Rename Columns (Standardization)
            # We map specific columns and fallback to lowercase for others
            df.rename(columns=rename_map, inplace=True)

            # Lowercase anything we missed to be safe
            df.columns = [c.lower() for c in df.columns]

            # 3. Process Dates
            # Note: We process dates AFTER renaming, so we look for the NEW name
            if date_cols:
                for col_name, fmt in date_cols.items():
                    if col_name in df.columns:
                        if fmt:
                            df[col_name] = pd.to_datetime(
                                df[col_name], format=fmt, errors="coerce"
                            )
                        else:
                            df[col_name] = pd.to_datetime(df[col_name], errors="coerce")

            # 4. Save
            df.to_sql(table_name, con, if_exists="replace", index=False)
            df.to_csv(self.processed_dir / f"{table_name}.csv", index=False)
            logger.info(f" Created {table_name} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"❌ Failed {table_name}: {e}")

    def process_and_load(self) -> None:
        """Step 3: Define schema mappings and execute load."""
        logger.info("Step 3: Processing and Standardizing Data...")
        self.csv_dir = self.find_raw_csv_directory()
        con = sqlite3.connect(self.db_path)

        # ---------------------------------------------------------
        # 1. DAILY ACTIVITY
        # ---------------------------------------------------------
        self._process_single_table(
            con,
            csv_name="dailyActivity_merged.csv",
            table_name="daily_activity",
            rename_map={
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
            date_cols={"event_date": "%m/%d/%Y"},
        )

        # ---------------------------------------------------------
        # 2. SLEEP DAY
        # ---------------------------------------------------------
        self._process_single_table(
            con,
            csv_name="sleepDay_merged.csv",
            table_name="sleep_day",
            rename_map={
                "Id": "user_id",
                "SleepDay": "event_date",  # Renaming SleepDay -> event_date for consistency
                "TotalSleepRecords": "sleep_records",
                "TotalMinutesAsleep": "minutes_asleep",
                "TotalTimeInBed": "time_in_bed",
            },
            date_cols={
                "event_date": None
            },  # Pandas usually auto-detects this format well
        )

        # ---------------------------------------------------------
        # 3. HEART RATE
        # ---------------------------------------------------------
        self._process_single_table(
            con,
            csv_name="heartrate_seconds_merged.csv",
            table_name="heartrate",
            rename_map={"Id": "user_id", "Time": "event_time", "Value": "bpm"},
            date_cols={"event_time": None},
        )

        # ---------------------------------------------------------
        # 4. HOURLY STEPS
        # ---------------------------------------------------------
        self._process_single_table(
            con,
            csv_name="hourlySteps_merged.csv",
            table_name="hourly_steps",
            rename_map={
                "Id": "user_id",
                "ActivityHour": "event_time",
                "StepTotal": "steps",
            },
            date_cols={"event_time": None},
        )

        # ---------------------------------------------------------
        # 5. WEIGHT LOG
        # ---------------------------------------------------------
        self._process_single_table(
            con,
            csv_name="weightLogInfo_merged.csv",
            table_name="weight_log",
            rename_map={
                "Id": "user_id",
                "Date": "event_time",  # Contains time, so event_time
                "WeightKg": "weight_kg",
                "WeightPounds": "weight_lbs",
                "BMI": "bmi",
            },
            date_cols={"event_time": None},
        )

        con.close()
        logger.info("Processing complete.")

    def run_full_pipeline(self) -> None:
        try:
            self.download_dataset()
            self.initialize_storage()
            self.process_and_load()
            logger.info("Pipeline execution finished successfully!")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    pipeline = FitbitDataPipeline()
    pipeline.run_full_pipeline()
