import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    cursor = con.cursor()
    cursor.execute(
        """
        SELECT count(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    )
    return cursor.fetchone()[0] > 0


def fetch_one(con: sqlite3.Connection, sql: str, params=None):
    if params is None:
        params = []
    cursor = con.cursor()
    cursor.execute(sql, params)
    return cursor.fetchone()


def compute_activity_baselines(
        con: sqlite3.Connection, user_id: int, window_days: int
) -> Dict[str, Optional[float]]:
    """
    Uses daily_activity table created by the ingestion pipeline.
    """
    if not table_exists(con, "daily_activity"):
        return {
            "avg_steps_per_day": None,
            "avg_calories_per_day": None,
            "avg_very_active_minutes_per_day": None,
            "avg_sedentary_minutes_per_day": None,
        }

    # SQLite doesn't strictly support "LIMIT ?" in subqueries in all versions/drivers easily
    # without ORDER BY, but commonly works.
    # We select the most recent N days.
    row = fetch_one(
        con,
        """
        SELECT AVG(total_steps)         AS avg_steps_per_day,
               AVG(calories)            AS avg_calories_per_day,
               AVG(very_active_minutes) AS avg_very_active_minutes_per_day,
               AVG(sedentary_minutes)   AS avg_sedentary_minutes_per_day
        FROM (SELECT *
              FROM daily_activity
              WHERE user_id = ?
              ORDER BY event_date DESC LIMIT ?)
        """,
        (user_id, window_days),
    )
    if not row:
        return {
            "avg_steps_per_day": None,
            "avg_calories_per_day": None,
            "avg_very_active_minutes_per_day": None,
            "avg_sedentary_minutes_per_day": None,
        }

    return {
        "avg_steps_per_day": safe_float(row[0]),
        "avg_calories_per_day": safe_float(row[1]),
        "avg_very_active_minutes_per_day": safe_float(row[2]),
        "avg_sedentary_minutes_per_day": safe_float(row[3]),
    }


def compute_sleep_baselines(
        con: sqlite3.Connection, user_id: int, window_days: int
) -> Dict[str, Optional[float]]:
    """
    Uses sleep_day table.
    Columns: minutes_asleep, time_in_bed (renamed in ingestion pipeline).
    """
    if not table_exists(con, "sleep_day"):
        return {
            "avg_sleep_minutes_per_night": None,
            "avg_time_in_bed_minutes_per_night": None,
        }

    row = fetch_one(
        con,
        """
        SELECT AVG(minutes_asleep) AS avg_sleep_minutes_per_night,
               AVG(time_in_bed)    AS avg_time_in_bed_minutes_per_night
        FROM (SELECT *
              FROM sleep_day
              WHERE user_id = ?
              ORDER BY event_date DESC LIMIT ?)
        """,
        (user_id, window_days),
    )
    if not row:
        return {
            "avg_sleep_minutes_per_night": None,
            "avg_time_in_bed_minutes_per_night": None,
        }

    return {
        "avg_sleep_minutes_per_night": safe_float(row[0]),
        "avg_time_in_bed_minutes_per_night": safe_float(row[1]),
    }


def compute_hr_baselines(
        con: sqlite3.Connection, user_id: int, window_days: int
) -> Dict[str, Optional[float]]:
    """
    Uses heartrate table.
    Note: SQLite lacks a native QUANTILE function.
    We approximate 'resting heart rate' by taking the MIN(bpm) for each day,
    then averaging those minimums over the window.
    """
    if not table_exists(con, "heartrate"):
        return {
            "avg_resting_hr_bpm": None,
            "avg_hr_bpm": None,
        }

    # 1. Identify the recent days window
    # We use a subquery to find the distinct dates first

    row = fetch_one(
        con,
        """
        WITH recent_dates AS (SELECT DISTINCT date (event_time) as d
        FROM heartrate
        WHERE user_id = ?
        ORDER BY d DESC
            LIMIT ?
            ),
            daily_stats AS (
        SELECT
            date (event_time) as d, MIN (bpm) as min_bpm, AVG (bpm) as avg_bpm
        FROM heartrate
        WHERE user_id = ?
          AND date (event_time) IN (SELECT d FROM recent_dates)
        GROUP BY 1
            )
        SELECT AVG(min_bpm), -- Proxy for resting HR
               AVG(avg_bpm)  -- Average HR
        FROM daily_stats
        """,
        (user_id, window_days, user_id),
    )

    if not row:
        return {
            "avg_resting_hr_bpm": None,
            "avg_hr_bpm": None,
        }

    return {
        "avg_resting_hr_bpm": safe_float(row[0]),
        "avg_hr_bpm": safe_float(row[1]),
    }


def compute_weight_metrics(
        con: sqlite3.Connection, user_id: int
) -> Dict[str, Optional[Any]]:
    """
    Uses weight_log table.
    """
    if not table_exists(con, "weight_log"):
        return {
            "weight_kg": None,
            "weight_lbs": None,
            "bmi": None,
            "weight_last_updated_iso": None,
        }

    row = fetch_one(
        con,
        """
        SELECT weight_kg,
               weight_lbs,
               bmi,
               event_time
        FROM weight_log
        WHERE user_id = ?
        ORDER BY event_time DESC LIMIT 1
        """,
        (user_id,),
    )
    if not row:
        return {
            "weight_kg": None,
            "weight_lbs": None,
            "bmi": None,
            "weight_last_updated_iso": None,
        }

    event_date = row[3]
    if event_date is None:
        ts = None
    else:
        ts = str(event_date)
        # Standardize strings if they look like dates without time
        if len(ts) == 10:
            ts = f"{ts}T00:00:00Z"
        elif " " in ts:
            # Handle SQLite default string format "YYYY-MM-DD HH:MM:SS"
            ts = ts.replace(" ", "T") + "Z"

    return {
        "weight_kg": safe_float(row[0]),
        "weight_lbs": safe_float(row[1]),
        "bmi": safe_float(row[2]),
        "weight_last_updated_iso": ts,
    }


def pick_activity_level(avg_steps_per_day: Optional[float]) -> Optional[str]:
    if avg_steps_per_day is None:
        return None
    if avg_steps_per_day < 5000:
        return "low"
    if avg_steps_per_day < 9000:
        return "moderate"
    return "high"


def build_user_profile(
        con: sqlite3.Connection,
        user_id: int,
        baseline_window_days: int,
        suggestiveness: float,
) -> Dict[str, Any]:
    activity = compute_activity_baselines(con, user_id, baseline_window_days)
    sleep = compute_sleep_baselines(con, user_id, baseline_window_days)
    hr = compute_hr_baselines(con, user_id, baseline_window_days)
    weight = compute_weight_metrics(con, user_id)

    profile: Dict[str, Any] = {
        "user_id": user_id,
        "demographics": {
            "age_years": 32,
            "sex": "male",
            "height_cm": 178.0,
        },
        "body_metrics": {
            "weight_kg": weight["weight_kg"],
            "weight_lbs": weight["weight_lbs"],
            "bmi": weight["bmi"],
            "body_fat_pct": None,
            "weight_last_updated_iso": weight["weight_last_updated_iso"],
        },
        "baselines": {
            "baseline_window_days": baseline_window_days,
            "avg_steps_per_day": activity["avg_steps_per_day"],
            "avg_calories_per_day": activity["avg_calories_per_day"],
            "avg_sleep_minutes_per_night": sleep["avg_sleep_minutes_per_night"],
            "avg_time_in_bed_minutes_per_night": sleep[
                "avg_time_in_bed_minutes_per_night"
            ],
            "avg_resting_hr_bpm": hr["avg_resting_hr_bpm"],
            "avg_hr_bpm": hr["avg_hr_bpm"],
            "avg_very_active_minutes_per_day": activity[
                "avg_very_active_minutes_per_day"
            ],
            "avg_sedentary_minutes_per_day": activity["avg_sedentary_minutes_per_day"],
            "hrv_ms": None,
        },
        "activity_profile": {
            "activity_level": pick_activity_level(activity["avg_steps_per_day"]),
            "preferred_workout_types": ["walking", "cycling"],
            "timezone": "Asia/Jerusalem",
        },
        "health_goals": {
            "daily_steps_goal": 10000,
            "weekly_active_minutes_goal": 150,
            "sleep_hours_goal": 7.5,
            "weight_goal_kg": 78.0,
        },
        "coaching_preferences": {
            "suggestiveness": float(suggestiveness),
            "tone": "supportive",
            "notification_frequency": "medium",
        },
        "system_state": {
            "onboarding_completed": True,
            "consent_medical_disclaimer": True,
            "last_interaction_iso": iso_now(),
            "last_suggestion_key": None,
        },
    }

    return profile


def main(user_id=1503960366):
    # Updated path to match SQLite pipeline output
    db_path = "dataset/db/fitbit.sqlite"

    baseline_window_days = 30
    suggestiveness = 0.7

    out_dir = Path("dataset/user_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{user_id}.json"

    if not Path(db_path).exists():
        raise RuntimeError(f"Database not found at {db_path}. Run the ingestion pipeline first.")

    con = sqlite3.connect(db_path)

    try:
        if not table_exists(con, "daily_activity"):
            raise RuntimeError(
                "Missing daily_activity table. Run your ingestion pipeline first to populate the database."
            )

        profile = build_user_profile(con, user_id, baseline_window_days, suggestiveness)
    finally:
        con.close()

    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote profile: {out_path}")


if __name__ == "__main__":
    user_ids = [1503960366, 2022484408]
    for user_id in user_ids:
        main(user_id)