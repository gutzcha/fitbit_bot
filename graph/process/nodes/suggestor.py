"""
graph/process/nodes/suggestor.py

The Suggestor Node adds personalized coaching 'nudges' to the response.
"""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from graph.process.agents.suggestor import build_suggestor_agent


# -----------------------------------------------------------------------------
# MOCK MEMORY SERVICE
# -----------------------------------------------------------------------------

def retrieve_memories(user_id: int, query: str) -> str:
    """
    Mock retrieval of user tendencies.
    In production, this would vector search your 'Memory' store.
    """
    # Example logic: Return specific tendencies that might affect coaching
    return (
        "- User prefers outdoor activities (walking, hiking) over gym.\n"
        "- User is trying to improve sleep consistency but struggles with late nights.\n"
        "- User responds well to positive reinforcement (""You're doing great!"").\n"
        "- User dislikes strict calorie counting."
    )


# -----------------------------------------------------------------------------
# NODE FACTORY
# -----------------------------------------------------------------------------

def make_suggestor_node(config: Dict[str, Any]):
    """
    Factory that returns the executable suggestor_node function.
    """
    provider = config.get("provider", "anthropic")
    model_type = config.get("model_type", "fast")

    # 1. Build the Agent Chain
    agent_chain = build_suggestor_agent(provider=provider, model_type=model_type)

    def suggestor_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggestor Node Logic:
        1. Checks validity (no errors/clarifications).
        2. Checks User Config (Suggestiveness threshold).
        3. Retrieves Memories.
        4. Generates Suggestion and appends to response.
        """
        # --- 1. Gatekeeping ---

        # If the execution step failed or needs clarification, skip coaching.
        if state.get("needs_clarification", False):
            return {}

        user_profile = state.get("user_profile")
        messages = state.get("messages", [])

        # We need the last answer to append to, and the user's question for context
        last_ai_message = messages[-1] if messages else None
        user_query = messages[-2].content if len(messages) >= 2 else "User Status Update"

        if not user_profile or not last_ai_message:
            return {}

        # --- 2. Check Preferences ---

        # Handle Pydantic model or dict
        prefs = getattr(user_profile, "coaching_preferences", {})
        if hasattr(prefs, "suggestiveness"):
            suggestiveness = prefs.suggestiveness
            tone = prefs.tone
        else:
            suggestiveness = prefs.get("suggestiveness", 0.5)
            tone = prefs.get("tone", "supportive")

        # Threshold check: If suggestiveness < 0.3, user wants raw data only.
        if suggestiveness < 0.3:
            return {}

        # --- 3. Prepare Context ---

        user_id = getattr(user_profile, "user_id", 0)
        relevant_memories = retrieve_memories(user_id, user_query)

        goals_summary = ""
        if hasattr(user_profile, "health_goals"):
            goals_summary = str(user_profile.health_goals)

        # Build the context string for the prompt
        input_context = (
            f"--- USER CONTEXT ---\n"
            f"Goals: {goals_summary}\n"
            f"Memories/Tendencies: {relevant_memories}\n\n"
            f"--- INTERACTION ---\n"
            f"User Asked: {user_query}\n"
            f"Data Agent Answered: {last_ai_message.content}\n\n"
            f"--- INSTRUCTION ---\n"
            f"Provide a suggestion based on the 'Decision Logic' in your system prompt."
        )

        # --- 4. Invoke Agent ---

        # Pass 'tone' for the system prompt variable and 'input_context' for the human part
        suggestion_text = agent_chain.invoke({
            "tone": tone,
            "input_context": input_context
        })

        # --- 5. Update State ---

        if not suggestion_text.strip():
            return {}

        # Create a new combined message.
        # We use Italics for the suggestion to visually separate it.
        new_content = f"{last_ai_message.content}\n\n_{suggestion_text}_"
        updated_message = AIMessage(content=new_content)

        return {
            # Replacing the last message effectively updates the conversation
            "messages": [updated_message],
            "response": new_content,
            "suggestion_included": True
        }

    return suggestor_node