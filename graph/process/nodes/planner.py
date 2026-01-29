# graph/process/nodes/planner.py
"""
PROCESS NODE, PLANNER
=====================

Runs the planner agent and stores ProcessPlan into AssistantState.

Responsibilities:
- Build planner_input from state (current date, user message, intent metadata, conversation hints, USER PROFILE)
- Invoke the planner with structured output
- Enforce bounded retries to avoid infinite loops
- If the plan is invalid or underspecified, ask the planner to repair it
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from langchain_core.exceptions import OutputParserException

from graph.helpers import get_current_date
from graph.process.agents.planner import build_planner
from graph.process.schemas import ProcessPlan
from graph.state import AssistantState


def _extract_conversation_hints(state: AssistantState) -> Dict[str, Any]:
    conv = state.get("conversation_state")
    if conv is None:
        return {}

    return {
        "current_topic": getattr(conv, "current_topic", None),
        "mentioned_metrics": list(getattr(conv, "mentioned_metrics", []) or []),
        "prior_intent": getattr(conv, "prior_intent", None),
    }


def _get_user_text(state: AssistantState) -> str:
    conv = state.get("conversation_state")
    if conv is not None:
        explicit = getattr(conv, "user_explicitly_asked", None)
        if explicit:
            return explicit
    return state["messages"][-1].content


def _validate_plan(plan: ProcessPlan) -> List[str]:
    """
    Returns a list of violations. This function does not suggest fixes or text.
    """
    violations: List[str] = []

    if plan.needs_clarification and not plan.clarification_question:
        violations.append(
            "needs_clarification=true but clarification_question is missing"
        )

    if not plan.selected_sources:
        violations.append("selected_sources is empty")

    has_metric = bool(plan.metrics)
    has_time = (
            plan.time_range is not None
            and plan.time_range.start_date is not None
            and plan.time_range.end_date is not None
    )

    # If the plan is not asking for clarification, it must be executable
    if not plan.needs_clarification:
        if not has_metric:
            violations.append("metrics is empty but needs_clarification=false")
        # Only enforce time range when the plan intends to fetch user metrics
        if "USER_METRICS" in plan.selected_sources and not has_time:
            violations.append(
                "time_range is missing or incomplete for USER_METRICS execution"
            )

    return violations


def _build_planner_input(state: AssistantState) -> str:
    """
    Constructs the context prompt for the Planner.
    Now includes User Profile to ensure goals and baselines are considered.
    """
    current_date = get_current_date()
    intent_meta = state.get("intent_metadata")
    hints = _extract_conversation_hints(state)
    user_text = _get_user_text(state)

    # ✅ NEW: Inject User Profile
    user_profile = state.get("user_profile")
    profile_str = "No user profile available."

    if user_profile:
        # Handle various object types safely
        if hasattr(user_profile, "model_dump_json"):
            profile_str = user_profile.model_dump_json()
        elif isinstance(user_profile, dict):
            profile_str = json.dumps(user_profile)
        else:
            profile_str = str(user_profile)

    return (
        f"Current date: {current_date}\n\n"
        f"User message: {user_text}\n\n"
        f"--- USER PROFILE CONTEXT ---\n{profile_str}\n\n"  # Injected here
        f"Intent metadata: {intent_meta.model_dump() if intent_meta else None}\n"
        f"Conversation hints: {hints}\n"
    )


def _build_repair_input(
        base_input: str, prior_plan: ProcessPlan | None, violations: List[str]
) -> str:
    return (
            f"{base_input}\n\n"
            f"Validation violations:\n- " + "\n- ".join(violations) + "\n\n"
                                                                      f"Repair task:\n"
                                                                      f"- Return a corrected ProcessPlan\n"
                                                                      f"- If anything essential is missing, set needs_clarification=true and provide exactly one clarification_question\n"
                                                                      f"- Confidence must be in [0, 1]\n\n"
                                                                      f"Previous plan (if any): {prior_plan.model_dump() if prior_plan else None}\n"
    )


def make_planner(config: Dict[str, Any]):
    provider = config.get("provider", "ollama")
    max_attempts = int(config.get("planner_max_attempts", 2))

    planner = build_planner(provider=provider, model_type="slow")

    def planner_node(state: AssistantState) -> Dict[str, Any]:
        attempts = int(state.get("planner_attempts", 0))
        base_input = _build_planner_input(state)

        last_plan: ProcessPlan | None = None
        last_violations: List[str] = []

        while attempts < max_attempts:
            try:
                if last_violations:
                    planner_input = _build_repair_input(
                        base_input, last_plan, last_violations
                    )
                else:
                    planner_input = base_input

                plan = planner.invoke({"planner_input": planner_input})
                violations = _validate_plan(plan)

                if not violations:
                    return {
                        "process_plan": plan,
                        "needs_clarification": bool(plan.needs_clarification),
                        "planner_attempts": attempts + 1,
                    }

                last_plan = plan
                last_violations = violations
                attempts += 1

            except OutputParserException:
                # The model produced invalid structured output, retry with repair framing
                last_plan = None
                last_violations = [
                    "structured output parse failure, return a valid ProcessPlan"
                ]
                attempts += 1

        # Final fallback: force the planner to produce a clarification question via repair framing
        final_input = _build_repair_input(
            base_input,
            last_plan,
            last_violations
            or [
                "retry limit reached, produce a clarification question that will unblock execution"
            ],
        )
        final_plan = planner.invoke({"planner_input": final_input})

        # If even here it violates the contract, we still return it, router will go to clarification
        # and you will see it in logs and tests
        return {
            "process_plan": final_plan,
            "needs_clarification": True,
            "planner_attempts": max_attempts,
        }

    return planner_node