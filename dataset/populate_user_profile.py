"""
Populates user profiles by combining:
1. Static configuration (demographics, goals) passed in at runtime.
2. Dynamic baselines computed from the SQLite database.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from graph.consts import DB_PATH, MOCK_USER_CONFIGS, PROFILE_DIR
from graph.helpers import get_current_date

# --- Helper Functions ---


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
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone()[0] > 0


def fetch_one(con: sqlite3.Connection, sql: str, params=None):
    if params is None:
        params = []
    cursor = con.cursor()
    cursor.execute(sql, params)
    return cursor.fetchone()


# --- Normalization Logic (CRITICAL PART) ---


def normalize_demographics(d: Any) -> Dict[str, Any]:
    """
    Converts static demographics into schema-compliant format.

    Schema expects:
    - age_years
    - sex
    - height_cm
    """
    if not isinstance(d, dict):
        return {}

    age_years = d.get("age_years", d.get("age"))
    height_cm = d.get("height_cm", d.get("height"))

    out: Dict[str, Any] = {
        "age_years": int(age_years) if age_years is not None else None,
        "sex": d.get("sex"),
        "height_cm": float(height_cm) if height_cm is not None else None,
    }

    return {k: v for k, v in out.items() if v is not None}


def normalize_coaching_preferences(p: Any) -> Dict[str, Any]:
    """
    Converts coaching preferences into schema-compliant format.

    Schema expects:
    - suggestiveness: float
    - tone: optional str
    """
    if not isinstance(p, dict):
        return {}

    mapping = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "very_high": 0.9,
    }

    raw = p.get("suggestiveness")
    suggestiveness: Optional[float] = None

    if isinstance(raw, (int, float)):
        suggestiveness = float(raw)
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if s in mapping:
            suggestiveness = mapping[s]
        else:
            try:
                suggestiveness = float(s)
            except Exception:
                suggestiveness = None

    out: Dict[str, Any] = {}
    if suggestiveness is not None:
        out["suggestiveness"] = max(0.0, min(1.0, suggestiveness))

    if "tone" in p and p["tone"] is not None:
        out["tone"] = str(p["tone"])

    return out


# --- Computation Logic ---


def compute_activity_baselines(
    con: sqlite3.Connection, user_id: int, window_days: int
) -> Dict[str, Optional[float]]:
    if not table_exists(con, "daily_activity"):
        return {}

    row = fetch_one(
        con,
        """
        SELECT AVG(total_steps), AVG(calories), AVG(very_active_minutes), AVG(sedentary_minutes)
        FROM (SELECT *
              FROM daily_activity
              WHERE user_id = ?
              ORDER BY event_date DESC LIMIT ?)
        """,
        (user_id, window_days),
    )
    if not row:
        return {}

    return {
        "avg_steps_per_day": safe_float(row[0]),
        "avg_calories_per_day": safe_float(row[1]),
        "avg_very_active_minutes_per_day": safe_float(row[2]),
        "avg_sedentary_minutes_per_day": safe_float(row[3]),
    }


def compute_sleep_baselines(
    con: sqlite3.Connection, user_id: int, window_days: int
) -> Dict[str, Optional[float]]:
    if not table_exists(con, "sleep_day"):
        return {}

    row = fetch_one(
        con,
        """
        SELECT AVG(minutes_asleep), AVG(time_in_bed)
        FROM (SELECT *
              FROM sleep_day
              WHERE user_id = ?
              ORDER BY event_date DESC LIMIT ?)
        """,
        (user_id, window_days),
    )
    if not row:
        return {}

    return {
        "avg_sleep_minutes_per_night": safe_float(row[0]),
        "avg_time_in_bed_minutes_per_night": safe_float(row[1]),
    }


def compute_hr_baselines(
    con: sqlite3.Connection, user_id: int, window_days: int
) -> Dict[str, Optional[float]]:
    if not table_exists(con, "heartrate"):
        return {}

    row = fetch_one(
        con,
        """
        WITH recent_dates AS (
            SELECT DISTINCT date(event_time) as d
            FROM heartrate
            WHERE user_id = ?
            ORDER BY d DESC LIMIT ?
        ),
        daily_stats AS (
            SELECT date(event_time) as d, MIN(bpm) as min_bpm, AVG(bpm) as avg_bpm
            FROM heartrate
            WHERE user_id = ? AND date(event_time) IN (SELECT d FROM recent_dates)
            GROUP BY 1
        )
        SELECT AVG(min_bpm), AVG(avg_bpm)
        FROM daily_stats
        """,
        (user_id, window_days, user_id),
    )

    if not row:
        return {}

    return {
        "avg_resting_hr_bpm": safe_float(row[0]),
        "avg_hr_bpm": safe_float(row[1]),
    }


def compute_weight_metrics(con: sqlite3.Connection, user_id: int) -> Dict[str, Any]:
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
        SELECT weight_kg, weight_lbs, bmi, event_time
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

    ts = str(row[3]) if row[3] else None
    if ts and len(ts) == 10:
        ts = f"{ts}T00:00:00Z"
    elif ts and " " in ts:
        ts = ts.replace(" ", "T") + "Z"

    return {
        "weight_kg": safe_float(row[0]),
        "weight_lbs": safe_float(row[1]),
        "bmi": safe_float(row[2]),
        "weight_last_updated_iso": ts,
    }


def pick_activity_level(avg_steps: Optional[float]) -> str:
    if avg_steps is None:
        return "unknown"
    if avg_steps < 5000:
        return "low"
    if avg_steps < 9000:
        return "moderate"
    return "high"


# --- Main Profile Builder ---


def build_user_profile(
    con: sqlite3.Connection,
    user_id: int,
    static_user_data: Dict[str, Any],
    baseline_window_days: int = 30,
) -> Dict[str, Any]:
    activity = compute_activity_baselines(con, user_id, baseline_window_days)
    sleep = compute_sleep_baselines(con, user_id, baseline_window_days)
    hr = compute_hr_baselines(con, user_id, baseline_window_days)
    weight = compute_weight_metrics(con, user_id)

    demographics = normalize_demographics(static_user_data.get("demographics", {}))
    coaching_preferences = normalize_coaching_preferences(
        static_user_data.get("coaching_preferences", {})
    )

    profile = {
        "user_id": user_id,
        "user_name": "Karl Kal",
        "demographics": demographics,
        "body_metrics": {
            "weight_kg": weight.get("weight_kg")
            or static_user_data.get("initial_weight_kg"),
            "weight_lbs": weight.get("weight_lbs"),
            "bmi": weight.get("bmi"),
            "body_fat_pct": static_user_data.get("body_fat_pct"),
            "weight_last_updated_iso": weight.get("weight_last_updated_iso"),
        },
        "baselines": {
            "baseline_window_days": baseline_window_days,
            "avg_steps_per_day": activity.get("avg_steps_per_day"),
            "avg_calories_per_day": activity.get("avg_calories_per_day"),
            "avg_sleep_minutes_per_night": sleep.get("avg_sleep_minutes_per_night"),
            "avg_time_in_bed_minutes_per_night": sleep.get(
                "avg_time_in_bed_minutes_per_night"
            ),
            "avg_resting_hr_bpm": hr.get("avg_resting_hr_bpm"),
            "avg_hr_bpm": hr.get("avg_hr_bpm"),
            "avg_very_active_minutes_per_day": activity.get(
                "avg_very_active_minutes_per_day"
            ),
            "avg_sedentary_minutes_per_day": activity.get(
                "avg_sedentary_minutes_per_day"
            ),
        },
        "activity_profile": {
            "activity_level": pick_activity_level(activity.get("avg_steps_per_day")),
            "preferred_workout_types": static_user_data.get(
                "preferred_workout_types", []
            ),
            "timezone": static_user_data.get("timezone", "UTC"),
        },
        "health_goals": static_user_data.get("health_goals", {}),
        "coaching_preferences": coaching_preferences,
        "system_state": {
            "onboarding_completed": True,
            "last_interaction_iso": get_current_date(),
            **static_user_data.get("system_state", {}),
        },
    }

    return profile


def populate_user_data(user_id: int, static_config: Dict[str, Any]):
    profile_dir = Path(PROFILE_DIR)
    profile_dir.mkdir(parents=True, exist_ok=True)
    out_path = profile_dir / f"{user_id}.json"

    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}. Run ingestion first.")

    con = sqlite3.connect(DB_PATH)
    try:
        profile = build_user_profile(con, user_id, static_user_data=static_config)
    finally:
        con.close()

    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote profile for {user_id} to: {out_path}")


def generate_all_user_profiles():
    for user_id, user_config in MOCK_USER_CONFIGS.items():
        populate_user_data(user_id, user_config)


if __name__ == "__main__":
    generate_all_user_profiles()
