"""
app/app.py

Streamlit interface for the Fitbit Conversational AI.
"""

import os
import sys

# ---------------- PATH SETUP ----------------

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from graph.consts import ENV_PATH
from dotenv import load_dotenv

is_loaded = load_dotenv(ENV_PATH, override=True)

print("\n--- ENVIRONMENT DEBUG ---")
print(f"Project Root:      {project_root}")
print(f"Looking for .env:  {ENV_PATH}")
print(f"File Found/Loaded: {is_loaded}")
print(f"LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING', 'Not Set')}")
print(
    f"LANGSMITH_API_KEY: {'[SET]' if os.environ.get('LANGSMITH_API_KEY') else '[MISSING]'}"
)
print("------------------------\n")

# ---------------- IMPORTS ----------------

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph.consts import CHAT_CONFIG_PATH
from graph.graph import build_graph
from graph.helpers import load_config
from graph.memory import MemoryManager

DEFAULT_CURRENT_USER_ID = 1503960366


# ---------------- HELPERS ----------------

def save_config(config: Dict[str, Any]) -> None:
    try:
        with open(CHAT_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save config: {e}")


def convert_ui_msgs_to_langchain(ui_messages: List[Dict[str, str]]):
    lc_messages = []
    for msg in ui_messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def rebuild_graph(config: Dict[str, Any]) -> None:
    st.session_state.graph = build_graph(config)
    st.session_state.config = config

    try:
        st.session_state.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    except Exception:
        pass


def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "config" not in st.session_state:
        st.session_state.config = load_config(CHAT_CONFIG_PATH)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0

    if "debug_trace" not in st.session_state:
        st.session_state.debug_trace = []

    if "graph" not in st.session_state:
        rebuild_graph(st.session_state.config)

    if "memory_manager" not in st.session_state:
        user_id = DEFAULT_CURRENT_USER_ID
        if isinstance(st.session_state.config, dict):
            user_id = st.session_state.config.get("user_id", user_id)

        st.session_state.memory_manager = MemoryManager(user_id=user_id)
        st.session_state.user_profile = st.session_state.memory_manager.load_user_profile()


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.session_state.debug_trace = []
    st.session_state.thread_id = str(uuid.uuid4())


def get_runtime_node_cfg(config: Dict[str, Any], path: str) -> Dict[str, Any]:
    runtime_nodes = config.get("runtime_nodes", {})
    node_cfg = runtime_nodes.get(path, {})
    return node_cfg if isinstance(node_cfg, dict) else {}


def set_runtime_node_cfg(config: Dict[str, Any], path: str, node_cfg: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    runtime_nodes = dict(config.get("runtime_nodes", {}))
    runtime_nodes[path] = node_cfg
    config["runtime_nodes"] = runtime_nodes
    return config


def _extract_response_and_suggestion(node_name: str, state_update: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (response_text, suggestion_text).

    Contract:
    - "response_text" is the main answer and should be shown even if no suggestion exists.
    - "suggestion_text" is only the coaching nudge. It should be appended after the response.

    This function is defensive because different nodes might emit different shapes.
    """
    response_text: Optional[str] = None
    suggestion_text: Optional[str] = None

    # 1. Direct response field
    resp = state_update.get("response")
    if isinstance(resp, str) and resp.strip():
        response_text = resp.strip()

    # 2. Suggestor node may include a suggestion explicitly
    sugg = state_update.get("suggestion")
    if isinstance(sugg, str) and sugg.strip():
        suggestion_text = sugg.strip()

    # 3. Some implementations only expose a combined "response" from suggestor.
    # If we have a known marker, try to split. Otherwise do not guess.
    # Here we do not split automatically to avoid corrupting content.
    # The preferred way is: suggestor returns {"suggestion": "..."}.
    # If your suggestor currently returns combined content, it will still show correctly
    # because response_text will already include it.

    # 4. Fallback: pull from execution_result
    exec_res = state_update.get("execution_result")
    if exec_res is not None and response_text is None:
        if isinstance(exec_res, dict):
            r = exec_res.get("response") or exec_res.get("answer") or exec_res.get("output")
            if isinstance(r, str) and r.strip():
                response_text = r.strip()
        else:
            r = getattr(exec_res, "response", None) or getattr(exec_res, "answer", None) or getattr(exec_res, "output", None)
            if isinstance(r, str) and r.strip():
                response_text = r.strip()

    return response_text, suggestion_text


# ---------------- STREAMLIT SETUP ----------------

st.set_page_config(
    page_title="Fitbit AI Assistant",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("System Control")

    with st.form("config_form"):
        st.subheader("Runtime Nodes")

        cfg_intent = get_runtime_node_cfg(st.session_state.config, "graph.nodes.intent")
        cfg_exec = get_runtime_node_cfg(st.session_state.config, "graph.process.nodes.execution")
        cfg_suggestor = get_runtime_node_cfg(st.session_state.config, "graph.process.nodes.suggestor")

        intent_model = cfg_intent.get("llm", {}).get("model", "ollama:qwen3:8b")
        confidence_threshold = float(cfg_intent.get("confidence_threshold", 0.7))
        max_history_limit = int(cfg_exec.get("max_history_limit", 20))
        max_iterations = int(cfg_exec.get("max_iterations", 5))
        suggestor_enabled = bool(cfg_suggestor.get("enabled", True))

        intent_model_new = st.text_input("Intent model", value=intent_model)

        confidence_threshold_new = st.slider(
            "Intent confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=confidence_threshold,
        )

        max_history_limit_new = st.slider(
            "Execution max history",
            min_value=2,
            max_value=50,
            value=max_history_limit,
        )

        max_iterations_new = st.slider(
            "Execution max iterations",
            min_value=1,
            max_value=30,
            value=max_iterations,
        )

        suggestor_enabled_new = st.checkbox(
            "Enable suggestor",
            value=suggestor_enabled,
        )

        if st.form_submit_button("Apply and rebuild"):
            new_config = dict(st.session_state.config)

            cfg_intent_new = dict(cfg_intent)
            cfg_intent_new["llm"] = {"model": intent_model_new}
            cfg_intent_new["confidence_threshold"] = float(confidence_threshold_new)
            new_config = set_runtime_node_cfg(new_config, "graph.nodes.intent", cfg_intent_new)

            cfg_exec_new = dict(cfg_exec)
            cfg_exec_new["max_history_limit"] = int(max_history_limit_new)
            cfg_exec_new["max_iterations"] = int(max_iterations_new)
            new_config = set_runtime_node_cfg(new_config, "graph.process.nodes.execution", cfg_exec_new)

            cfg_suggestor_new = dict(cfg_suggestor)
            cfg_suggestor_new["enabled"] = bool(suggestor_enabled_new)
            new_config = set_runtime_node_cfg(new_config, "graph.process.nodes.suggestor", cfg_suggestor_new)

            save_config(new_config)
            rebuild_graph(new_config)
            st.success("Config applied.")
            st.rerun()

    st.divider()

    st.subheader("User profile")
    if st.session_state.user_profile:
        prof = st.session_state.user_profile
        with st.expander(f"User ID: {getattr(prof, 'user_id', 'unknown')}", expanded=False):
            if hasattr(prof, "model_dump"):
                st.json(prof.model_dump())
            else:
                st.json(prof)
    else:
        st.warning("Profile not loaded.")

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Turns", st.session_state.turn_count)
    col2.metric("Thread", st.session_state.thread_id[:6])

    if st.button("Reset conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

# ---------------- MAIN ----------------

st.title("Fitbit Health Assistant")

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("How are my steps looking today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    inputs = {
        "messages": convert_ui_msgs_to_langchain(st.session_state.messages),
        "user_profile": st.session_state.user_profile,
    }

    run_cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

    final_response: Optional[str] = None
    final_suggestion: Optional[str] = None
    debug_trace: List[str] = []

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        suggestion_placeholder = st.empty()
        status_container = st.status("Thinking...", expanded=True)

        try:
            for event in st.session_state.graph.stream(inputs, config=run_cfg):
                for node_name, state_update in event.items():
                    debug_trace.append(node_name)

                    if state_update is None:
                        status_container.write(f"{node_name}: no state update")
                        continue

                    if not isinstance(state_update, dict):
                        status_container.write(f"{node_name}: non-dict update: {type(state_update)}")
                        continue

                    if node_name == "INTENT":
                        intent_data = state_update.get("intent_metadata")
                        if intent_data:
                            intent_label = (
                                intent_data.get("intent", "UNKNOWN")
                                if isinstance(intent_data, dict)
                                else getattr(intent_data, "intent", "UNKNOWN")
                            )
                            status_container.write(f"Intent: {intent_label}")

                    elif node_name == "PROCESS":
                        status_container.write("Process: running execution agent")

                    elif node_name == "SUGGESTOR":
                        status_container.write("Suggestor: evaluating")

                    resp_text, sugg_text = _extract_response_and_suggestion(node_name, state_update)

                    # Always show the main response as soon as we can
                    if isinstance(resp_text, str) and resp_text.strip():
                        final_response = resp_text
                        message_placeholder.markdown(final_response)

                    # Suggestion should be shown only after response
                    if isinstance(sugg_text, str) and sugg_text.strip():
                        final_suggestion = sugg_text
                        if final_response:
                            suggestion_placeholder.markdown(final_suggestion)

            status_container.update(label="Complete", state="complete", expanded=False)

            if final_response:
                combined = final_response
                if final_suggestion:
                    combined = f"{final_response}\n\n{final_suggestion}"

                st.session_state.messages.append({"role": "assistant", "content": combined})
                st.session_state.turn_count += 1
                st.session_state.debug_trace = debug_trace
            else:
                message_placeholder.error("The agent completed but returned no response.")

        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"Execution failed: {str(e)}")
            import traceback
            traceback.print_exc()

if st.session_state.debug_trace:
    with st.expander("Trace and state debugger"):
        st.write("Execution path: " + " -> ".join(st.session_state.debug_trace))

        try:
            current_state = st.session_state.graph.get_state(run_cfg)
            st.json(current_state.values)
        except Exception:
            st.caption("State snapshot unavailable.")
