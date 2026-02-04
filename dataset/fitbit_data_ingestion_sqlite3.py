"""
# fitbit_data_ingestion_sqlite3.py

Fitbit Data Ingestion Pipeline.
Downloading, cleaning, and loading data based on
configurations defined in graph.consts and graph.dataset_schema.

This module orchestrates the ETL (Extract, Transform, Load) process:

1.  **Extract**: Downloads the raw dataset from Kaggle (using IDs from `consts.py`)
    and dynamically locates the raw CSV files within nested subdirectories.

2.  **Transform**: Applies the processing rules defined in `dataset_config.py`.
    -   Renames columns to standardized snake_case.
    -   Parses disparate date strings into uniform datetime objects.
    -   Filters only the tables defined in the schema mapping.

3.  **Load**: Persists the standardized data into two destinations:
    -   A SQLite database (`fitbit.sqlite`) for application use.
    -   A `clean/` directory containing processed CSVs for inspection.
"""

import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd

# --- Imports from your refactored modules ---
from graph.consts import (
    DATASET_DIR,
    DB_PATH,
    KAGGLE_DATASET_ID,
    PROCESSED_DIR,
    RAW_DATA_SUBDIRS,
)
from dataset.dataset_config import SCHEMA_MAPPING, TableConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FitbitDataPipeline:
    def __init__(self):
        # Paths are pulled from consts.py
        self.dataset_dir = DATASET_DIR
        self.processed_dir = PROCESSED_DIR
        self.db_path = DB_PATH
        self.csv_dir: Optional[Path] = None

    def download_dataset(self) -> None:
        """Step 1: Download Fitbit dataset from Kaggle."""
        logger.info("Step 1: Downloading dataset from Kaggle...")
        src = kagglehub.dataset_download(KAGGLE_DATASET_ID)

        # Standardize the download location by looking for the expected subdir
        target_path = self.dataset_dir / RAW_DATA_SUBDIRS[0]

        # Only copy if it doesn't already exist or if you want to force overwrite
        if not target_path.exists():
            try:
                shutil.copytree(src, self.dataset_dir, dirs_exist_ok=True)
                logger.info(f"Dataset copied to: {self.dataset_dir}")
            except Exception as e:
                logger.error(f"Failed to copy dataset: {e}")
        else:
            logger.info("Dataset already exists at target location.")

    def initialize_storage(self) -> None:
        """Step 2: Create directories for DB and processed CSVs."""
        logger.info("Step 2: Initializing storage directories...")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Reset DB to ensure a clean slate
        if self.db_path.exists():
            self.db_path.unlink()
            logger.info("Existing database removed.")

        # Initialize empty DB file
        con = sqlite3.connect(self.db_path)
        con.close()

    def find_raw_csv_directory(self) -> Path:
        """Helper: Locate the directory containing the raw CSV files."""
        # 1. Check the specific subdirectory structure we expect from Kaggle
        current_path = self.dataset_dir
        for subdir in RAW_DATA_SUBDIRS:
            current_path = current_path / subdir

        if current_path.exists():
            return current_path

        # 2. Fallback: Search recursively if the folder structure changed
        logger.info("Standard path not found. Searching recursively...")
        for root, _, files in os.walk(self.dataset_dir):
            if any(f.endswith(".csv") for f in files):
                return Path(root)

        raise FileNotFoundError(f"Could not find raw CSV files in {self.dataset_dir}")

    def _process_single_table(
            self,
            con: sqlite3.Connection,
            table_name: str,
            config: TableConfig
    ):
        """
        Generic processor that reads config dicts and loads data to SQLite.
        """
        csv_name = config["csv_name"]
        rename_map = config["rename_map"]
        date_cols = config["date_cols"]

        file_path = self.csv_dir / csv_name
        if not file_path.exists():
            logger.warning(f"Skipping {table_name}: {csv_name} not found.")
            return

        logger.info(f"Processing {table_name}...")
        try:
            # 1. Read CSV
            df = pd.read_csv(file_path)

            # 2. Standardize Columns
            df.rename(columns=rename_map, inplace=True)
            df.columns = [c.lower() for c in df.columns]

            # 3. Process Dates
            if date_cols:
                for col_name, fmt in date_cols.items():
                    if col_name in df.columns:
                        if fmt:
                            df[col_name] = pd.to_datetime(
                                df[col_name], format=fmt, errors="coerce"
                            )
                        else:
                            df[col_name] = pd.to_datetime(df[col_name], errors="coerce")

            # 4. Save to DB and CSV
            df.to_sql(table_name, con, if_exists="replace", index=False)
            df.to_csv(self.processed_dir / f"{table_name}.csv", index=False)
            logger.info(f"-> Saved {table_name} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"❌ Failed {table_name}: {e}")

    def process_and_load(self) -> None:
        """Step 3: Iterate over the schema defined in dataset_config.py."""
        logger.info("Step 3: Processing and Standardizing Data...")
        self.csv_dir = self.find_raw_csv_directory()

        con = sqlite3.connect(self.db_path)

        # Loop through the schema mapping
        for table_name, config in SCHEMA_MAPPING.items():
            self._process_single_table(con, table_name, config)

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