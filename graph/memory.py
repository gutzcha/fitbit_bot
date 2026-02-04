"""
graph/memory.py

Unified Memory Manager.
Handles:
1. Short-Term: Context window trimming (Token management).
2. Long-Term: User Profile loading and Preference updates.
"""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate

from graph.consts import PROFILE_DIR
from graph.schemas import UserProfile

# ─────────────────────────────────────────────────────────────────────────────
# 1. SHORT-TERM MEMORY (Context Window)
# ─────────────────────────────────────────────────────────────────────────────


def trim_conversation_history(
    messages: List[BaseMessage], max_messages: int = 10
) -> List[BaseMessage]:
    """
    Truncates conversation history to fit within the model's context window.
    Always preserves the System Message and the most recent N messages.
    """
    return trim_messages(
        messages,
        max_tokens=max_messages,  # Using message count as proxy for tokens here
        strategy="last",
        token_counter=len,  # 1 message = 1 token (simplification)
        include_system=True,  # Critical: Never lose the System Prompt
        allow_partial=False,
        start_on="human",  # Don't cut in the middle of a Q&A pair
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. LONG-TERM MEMORY (User Profile & Preferences)
# ─────────────────────────────────────────────────────────────────────────────


class MemoryManager:
    """
    Controller for persistent user data (Profiles, Preferences).
    """

    def __init__(self, user_id: int, profile_dir=PROFILE_DIR):
        profile_dir = Path(profile_dir) if isinstance(profile_dir, str) else profile_dir
        self.user_id = user_id
        self.profile_path = Path(PROFILE_DIR) / f"{user_id}.json"

    def load_user_profile(self) -> Optional[UserProfile]:
        """
        Loads the rich UserProfile object (Demographics, Baselines, Goals).
        """
        if not self.profile_path.exists():
            return None

        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserProfile(**data)
        except Exception as e:
            print(f"Error loading profile for {self.user_id}: {e}")
            return None

    def update_preferences(
        self, messages: List[BaseMessage], llm: BaseChatModel
    ) -> bool:
        """
        DISABLED FOR NOW.
        Returns False immediately to prevent auto-updates.
        """
        return False
