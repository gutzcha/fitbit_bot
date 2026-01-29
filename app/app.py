"""
app/app.py

Streamlit interface for the Fitbit Conversational AI.
"""

import os
import sys
# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
# Construct absolute path to .env
env_path = os.path.join(project_root, ".env")

# Force reload with override=True to ensure .env takes precedence
is_loaded = load_dotenv(env_path, override=True)

print("\n--- 🔍 ENVIRONMENT DEBUG ---")
print(f"Project Root:      {project_root}")
print(f"Looking for .env:  {env_path}")
print(f"File Found/Loaded: {is_loaded}")
print(f"LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING', 'Not Set')}")
print(f"LANGSMITH_API_KEY:    {'[SET]' if os.environ.get('LANGSMITH_API_KEY') else '[MISSING]'}")
print("----------------------------\n")

import json


import uuid
from typing import Any, Dict, List

import streamlit as st
from graph.consts import CHAT_CONFIG_PATH



# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.messages import AIMessage, HumanMessage

from graph.graph import build_graph
from graph.helpers import load_config
from graph.memory import MemoryManager
from graph.schemas import UserProfile



# Mock User ID for this prototype
DEFAULT_CURRENT_USER_ID = 1503960366


# ─────────────────────────────────────────────────────────────────────────────
# SESSION & STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def save_config(config: Dict[str, Any]):
    """Save current config to JSON."""
    try:
        with open(CHAT_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save config: {e}")


def convert_ui_msgs_to_langchain(ui_messages: List[Dict[str, str]]):
    """Converts Streamlit dict messages to LangChain BaseMessage objects."""
    lc_messages = []
    for msg in ui_messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def rebuild_graph(config: Dict[str, Any]):
    """
    Forces a rebuild of the LangGraph object with the new configuration.
    This ensures the provider switch (Ollama <-> Anthropic) takes effect immediately.
    """
    st.session_state.graph = build_graph(config)
    st.session_state.config = config

    # Optional: Generate visualization
    try:
        st.session_state.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    except Exception:
        pass


def init_session():
    """Initialize session state variables."""
    # 1. Identity
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # 2. Configuration (Load once, then persist in session)
    if "config" not in st.session_state:
        st.session_state.config = load_config(CHAT_CONFIG_PATH)

    # 3. UI State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0

    if "debug_trace" not in st.session_state:
        st.session_state.debug_trace = []

    # 4. Graph Construction (Lazy Load or Re-Load if config changed externally)
    # We check if graph exists; if not, build it with current session config
    if "graph" not in st.session_state:
        rebuild_graph(st.session_state.config)

    # 5. Memory Manager & User Profile
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager(user_id=st.session_state.config.get("user_id", DEFAULT_CURRENT_USER_ID))
        # Load profile once and store in session
        st.session_state.user_profile = st.session_state.memory_manager.load_user_profile()


def reset_conversation():
    """Clears chat history but keeps configuration."""
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.session_state.debug_trace = []
    st.session_state.thread_id = str(uuid.uuid4())  # New thread ID = Clean MemorySaver context


# ─────────────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fitbit AI Assistant",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session()

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.title("⚙️ System Control")

    # 1. Live Configuration
    with st.form("config_form"):
        st.subheader("Model Settings")

        current_provider = st.session_state.config.get("provider", "ollama")

        new_provider = st.selectbox(
            "LLM Provider",
            ["ollama", "anthropic"],
            index=0 if current_provider == "ollama" else 1,
            help="Switching this rebuilds the entire graph."
        )

        use_fallback = st.checkbox(
            "Enable Slow Fallback",
            value=st.session_state.config.get("slow_fallback_enabled", True),
            help="Use a larger model if the fast one is unsure."
        )

        history_limit = st.slider(
            "Max History Context",
            min_value=2, max_value=20,
            value=st.session_state.config.get("max_history_limit", 10)
        )

        # Submit button triggers reload
        if st.form_submit_button("Apply & Rebuild Graph"):
            new_config = {
                "provider": new_provider,
                "slow_fallback_enabled": use_fallback,
                "max_history_limit": history_limit,
                "model_type": "slow"  # Defaulting to slow for robustness
            }
            save_config(new_config)
            rebuild_graph(new_config)
            st.success(f"Switched to {new_provider}!")
            st.rerun()

    st.divider()

    # 2. User Context (FULL VIEW)
    st.subheader("👤 User Profile")
    if st.session_state.user_profile:
        prof = st.session_state.user_profile
        with st.expander(f"User ID: {prof.user_id}", expanded=False):
            # Display the FULL JSON structure
            if hasattr(prof, "model_dump"):
                st.json(prof.model_dump())
            else:
                st.json(prof) # Fallback if not Pydantic v2
    else:
        st.warning("Profile not loaded.")

    st.divider()

    # 3. Session Metrics
    col1, col2 = st.columns(2)
    col1.metric("Turns", st.session_state.turn_count)
    col2.metric("Thread", st.session_state.thread_id[:6])

    if st.button("🗑️ Reset Conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

# --- MAIN AREA ---

st.title("⌚ Fitbit Health Assistant")
st.caption(f"Powered by **{st.session_state.config.get('provider', 'unknown').upper()}** | Graph Architecture")

# 1. Render Chat History
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 2. Chat Input Handler
if prompt := st.chat_input("How are my steps looking today?"):

    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # B. Prepare Inputs
    # IMPORTANT: We explicitly inject the User Profile into the state here
    # so the Planner/Executor can see it immediately.
    inputs = {
        "messages": convert_ui_msgs_to_langchain(st.session_state.messages),
        "user_profile": st.session_state.user_profile
    }

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    final_response = None
    debug_trace = []  # Capture the path taken

    # C. Execute Graph
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        status_container = st.status("Thinking...", expanded=True)

        try:
            # Stream events to show progress
            for event in st.session_state.graph.stream(inputs, config=config):

                for node_name, state_update in event.items():
                    debug_trace.append(node_name)

                    # Update Status Bar based on Node
                    if node_name == "INTENT":
                        intent_data = state_update.get("intent_metadata")
                        if intent_data:
                            # Handle Pydantic access safely
                            if isinstance(intent_data, dict):
                                intent_label = intent_data.get("intent", "UNKNOWN")
                            else:
                                intent_label = getattr(intent_data, "intent", "UNKNOWN")
                            status_container.write(f"🧠 Intent: **{intent_label}**")

                    elif node_name == "PLANNER":
                        plan = state_update.get("process_plan")
                        if plan:
                            steps = 0
                            if hasattr(plan, "steps"):
                                steps = len(plan.steps)
                            elif isinstance(plan, dict):
                                steps = len(plan.get("steps", []))
                            status_container.write(f"📋 Plan: Created **{steps} steps**")

                    elif node_name == "EXECUTION_AGENT":
                        status_container.write("⚙️ Executing Tools...")

                    elif node_name == "SUGGESTOR":
                        if state_update.get("suggestion_included"):
                            status_container.write("💡 Adding Coaching Suggestion...")

                    # Capture Final Response
                    if "response" in state_update and state_update["response"]:
                        final_response = state_update["response"]

            status_container.update(label="Complete", state="complete", expanded=False)

            # D. Render Final Response
            if final_response:
                message_placeholder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.session_state.turn_count += 1
                st.session_state.debug_trace = debug_trace
            else:
                message_placeholder.error("The agent completed but returned no response.")

        except Exception as e:
            status_container.update(label="Error", state="error")
            st.error(f"Graph Execution Failed: {str(e)}")
            import traceback
            traceback.print_exc()

# --- DEBUGGING FOOTER ---
if st.session_state.debug_trace:
    with st.expander("🔍 Trace & State Debugger"):
        st.write(f"**Execution Path:** {' → '.join(st.session_state.debug_trace)}")

        # Show specific details if available
        try:
            # Snapshot of current state
            current_state = st.session_state.graph.get_state(config)
            st.json(current_state.values)
        except Exception:
            st.caption("State snapshot unavailable (Thread ID may be new).")