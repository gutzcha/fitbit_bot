from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from graph.helpers import load_config
from graph.schemas import ConversationState, IntentMetadata, UserProfile

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "app" / "config.json"

SAMPLE_QUESTION = "how many steps did i take over the past week?"
SAMPLE_AI_ANSWER = (
    "You took a total of 97,573 steps over the past week (April 4-11, 2016), "
    "averaging about 12,196 steps per day. This is slightly above your daily "
    "goal of 12,000 steps. Your consistency with high-intensity workouts like "
    "running and swimming is paying off. Let me know if you'd like help tracking "
    "progress toward your weight goal too."
)


def load_runtime_config() -> dict:
    load_dotenv()
    return load_config(CONFIG_PATH)


def get_runtime_node(config: dict, key: str) -> dict:
    return (config.get("runtime_nodes", {}) or {}).get(key, {}) or {}


def build_llm(llm_config: dict):
    if not llm_config:
        raise ValueError("Missing LLM config")
    return init_chat_model(**llm_config)


def sample_conversation_state() -> ConversationState:
    return ConversationState(
        current_topic="activity",
        mentioned_metrics={"steps"},
        prior_intent="METRIC_RETRIEVAL",
        turn_count=1,
        user_explicitly_asked=SAMPLE_QUESTION,
    )


def sample_intent_metadata() -> IntentMetadata:
    return IntentMetadata(
        confidence=0.99,
        current_topic="activity",
        intent="METRIC_RETRIEVAL",
        is_followup=False,
        mentioned_metrics=["steps"],
        needs_clarification=False,
        response_type="DATA_LOOKUP",
        suggested_sources=["USER_METRICS"],
    )


def sample_user_profile() -> UserProfile:
    return UserProfile.model_validate(
        {
            "activity_profile": {
                "activity_level": "high",
                "preferred_workout_types": ["running", "swimming"],
                "timezone": "Asia/Jerusalem",
            },
            "baselines": {
                "avg_calories_per_day": 1796.2105263157894,
                "avg_sedentary_minutes_per_day": 809.8421052631579,
                "avg_steps_per_day": 11640.526315789473,
                "avg_very_active_minutes_per_day": 35.8421052631579,
                "baseline_window_days": 30,
            },
            "body_metrics": {
                "bmi": 22.9699993133545,
                "weight_kg": 53.2999992370605,
                "weight_last_updated_iso": "2016-04-05T23:59:59Z",
                "weight_lbs": 117.506384062611,
            },
            "coaching_preferences": {
                "suggestiveness": 0.8,
                "tone": "energetic",
            },
            "demographics": {
                "age_years": 32,
                "height_cm": 178,
                "sex": "male",
            },
            "health_goals": {
                "daily_steps_goal": 12000,
                "weight_goal_kg": 75,
            },
            "system_state": {
                "consent_medical_disclaimer": False,
                "last_interaction_iso": "2016-04-11",
                "onboarding_completed": True,
            },
            "user_id": 1503960366,
            "user_name": "Karl Kal",
        }
    )


def sample_messages():
    return [HumanMessage(content=SAMPLE_QUESTION)]


def sample_message_history():
    return [
        HumanMessage(content=SAMPLE_QUESTION),
        AIMessage(content=SAMPLE_AI_ANSWER),
    ]
