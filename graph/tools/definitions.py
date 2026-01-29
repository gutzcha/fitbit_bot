"""
graph/tools/definitions.py

A library of optimized, pre-defined SQL tools for the most common user queries.
Also includes the 'fallback' tool that generates custom SQL.
"""

import json
import os

from langchain_core.tools import tool

from graph.consts import PROFILE_DIR


@tool
def get_user_profile_json(user_id: int) -> str:
    """
    Retrieves the user's static profile (Goals, Weight, Height, Preferences) from JSON.
    Use this for "What is my step goal?" or "What is my current weight?".

    Args:
    user_id (int): Numeric Fitbit user identifier used to locate the profile file.

    Returns:
        str: JSON-formatted string containing goals, body metrics, and coaching preferences,
             or an explanatory error message if the profile is unavailable.

    """

    path = f"{PROFILE_DIR}/{user_id}.json"
    if not os.path.exists(path):
        return "Profile file not found."

    try:
        with open(path, "r") as f:
            data = json.load(f)
            # Flatten/Simplify for the LLM
            summary = {
                "goals": data.get("health_goals"),
                "body": data.get("body_metrics"),
                "preferences": data.get("coaching_preferences"),
            }
            return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error reading profile: {e}"
