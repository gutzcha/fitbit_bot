"""
graph/process/nodes/execution.py

The Execution Node is responsible for carrying out the ProcessPlan.
It acts as a ReAct agent that can call tools (SQL, RAG) and must return
a strictly structured response to ensure the graph state is updated correctly.
"""

from typing import Dict, Any, Optional
import json
from langchain_core.messages import AIMessage, HumanMessage

from graph.process.agents.execution import build_execution_agent
from graph.process.schemas import ExecutionResponse


def make_execution_agent(config: Dict[str, Any]):
    """
    Factory for the execution node.
    """
    provider = config.get("provider", "anthropic")
    model_type = config.get("model_type", "slow")

    # 1. Build the specific agent instance
    agent_runnable = build_execution_agent(provider=provider, model_type=model_type)

    def execution_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execution Node:
        1. Extracts User Profile and Plan from state.
        2. Injects them as context for the agent.
        3. Invokes agent and returns structured response.
        """
        process_plan = state.get("process_plan")
        user_profile = state.get("user_profile")

        # 2. Serialize Plan
        if hasattr(process_plan, "model_dump_json"):
            plan_text = process_plan.model_dump_json()
        else:
            plan_text = str(process_plan)

        # 3. Serialize User Profile
        # We explicitly inject this so the model never claims "I don't know your age"
        if hasattr(user_profile, "model_dump_json"):
            profile_text = user_profile.model_dump_json()
        elif isinstance(user_profile, dict):
            profile_text = json.dumps(user_profile, indent=2)
        else:
            profile_text = str(user_profile)

        # 4. Construct the Input Instruction
        context_message = HumanMessage(
            content=(
                f"--- GLOBAL USER CONTEXT ---\n"
                f"You have direct access to the user's profile (Age, Goals, Baselines). "
                f"USE THIS DATA if the plan implies a need for personal context (e.g. benchmarks).\n"
                f"```json\n{profile_text}\n```\n\n"
                f"--- EXECUTION PLAN ---\n"
                f"Follow these steps to fetch dynamic metrics (SQL) or general info (KB).\n"
                f"```json\n{plan_text}\n```\n\n"
                f"Task: Execute the plan and synthesize an answer using the Global Context + Tool Results."
            )
        )

        # Append to history
        input_messages = state["messages"] + [context_message]

        # 5. Invoke Agent
        result = agent_runnable.invoke({"messages": input_messages})

        # 6. Extract Structured Output
        structured_data: Optional[ExecutionResponse] = result.get("structured_response")

        # --- Fallback Logic ---
        if not structured_data:
            if "output" in result and result["output"]:
                structured_data = ExecutionResponse(
                    answer=str(result["output"]),
                    confidence=0.5,
                    needs_clarification=False
                )
            else:
                structured_data = ExecutionResponse(
                    answer="Error: Failed to generate a structured response.",
                    confidence=0.0,
                    needs_clarification=True
                )

        # 7. Return State Updates
        return {
            "messages": [AIMessage(content=structured_data.answer)],
            "response": structured_data.answer,
            "needs_clarification": structured_data.needs_clarification,
            "grounding_metadata": {
                "confidence": structured_data.confidence,
                "clarification_question": structured_data.clarification_question
            }
        }

    return execution_node