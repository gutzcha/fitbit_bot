# debug_run_process_node.py
# Run this from the project root:
#   (venv) python debug_run_process_node.py
#
# It will invoke graph.process.process.make_process_node(...) directly,
# feed it a minimal AssistantState dict, and print the returned update.

from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from graph.config_loader import load_graph_config
from graph.helpers import get_current_date
from graph.memory import MemoryManager
from graph.process.process import make_process_node

DEFAULT_USER_ID = 1503960366


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # Pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass

    # Dataclasses
    if is_dataclass(x):
        try:
            from dataclasses import asdict

            return asdict(x)
        except Exception:
            pass

    # LangChain messages
    if isinstance(x, BaseMessage):
        return {
            "type": x.type,
            "content": x.content,
            "additional_kwargs": getattr(x, "additional_kwargs", None),
        }

    return str(x)


def _print_update(update: Dict[str, Any]) -> None:
    print("\n================ PROCESS NODE UPDATE ================\n")

    resp = update.get("response")
    if isinstance(resp, str) and resp.strip():
        print("response:\n")
        print(resp.strip())
        print()

    needs = update.get("needs_clarification")
    if needs is not None:
        print(f"needs_clarification: {needs}")
    cq = update.get("clarification_question")
    if cq:
        print(f"clarification_question: {cq}")

    msgs = update.get("messages")
    if isinstance(msgs, list):
        print("\nmessages (tail):")
        for m in msgs[-6:]:
            if isinstance(m, HumanMessage):
                print(f"- HUMAN: {m.content}")
            elif isinstance(m, AIMessage):
                print(f"- AI: {m.content}")
            else:
                print(f"- {type(m).__name__}: {getattr(m, 'content', str(m))}")

    exec_res = update.get("execution_result")
    if exec_res is not None:
        print("\nexecution_result (json):")
        print(json.dumps(_to_jsonable(exec_res), indent=2))

    print("\n=====================================================\n")


def build_min_state(
    user_profile: Any,
    messages: List[BaseMessage],
    conversation_state: Optional[Any] = None,
    intent_metadata: Optional[Any] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "messages": messages,
        "user_profile": user_profile,
        "conversation_state": conversation_state,
        "intent_metadata": intent_metadata,
        "needs_clarification": False,
        "response": None,
    }

    # keep these optional
    if conversation_state is not None:
        state["conversation_state"] = conversation_state
    if intent_metadata is not None:
        state["intent_metadata"] = intent_metadata

    return state


def load_profile(user_id: int) -> Any:
    mm = MemoryManager(user_id=user_id)
    prof = mm.load_user_profile()
    if prof is None:
        raise RuntimeError(
            "User profile not found. Generate profiles first or check PROFILE_DIR."
        )
    return prof


def main() -> None:
    config = load_graph_config()
    process_node = make_process_node(config)

    user_id = config.get("user_id", DEFAULT_USER_ID)
    user_profile = load_profile(int(user_id))

    print("Manual process-node runner")
    print(f"User ID: {user_id}")
    print(f"Current Date (helper): {get_current_date()}")
    print("Type a question and press Enter. Empty input exits.\n")

    history: List[BaseMessage] = []

    while True:
        q = input("> ").strip()
        if not q:
            break

        history.append(HumanMessage(content=q))

        # Minimal AssistantState dict. If you have real intent/conv state objects,
        # pass them here too.
        state = build_min_state(
            user_profile=user_profile,
            messages=history,
            conversation_state=None,
            intent_metadata=None,
        )

        update = process_node(state)

        _print_update(update)

        # Important: keep conversation history consistent with what the node returns
        # so you can test followups.
        msgs = update.get("messages")
        if isinstance(msgs, list) and msgs:
            history = msgs
        else:
            # fallback: append the response if it exists
            resp = update.get("response")
            if isinstance(resp, str) and resp.strip():
                history.append(AIMessage(content=resp.strip()))


if __name__ == "__main__":
    main()
