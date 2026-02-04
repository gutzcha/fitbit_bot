"""
app/app.py

Streamlit interface for the Fitbit Conversational AI.
"""

import os
import sys
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# ---------------- PATH SETUP ----------------

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from graph.consts import ENV_PATH, PROFILE_DIR, CHAT_CONFIG_PATH
from graph.graph import build_graph
from graph.helpers import load_config
from graph.memory import MemoryManager

# Load env
is_loaded = load_dotenv(ENV_PATH, override=True)

DEFAULT_CURRENT_USER_ID = 1503960366


# ---------------- FILE AND CONFIG HELPERS ----------------


def save_config(config: Dict[str, Any]) -> None:
    try:
        with open(CHAT_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save config: {e}")


def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_list_user_ids(profile_dir: str) -> List[int]:
    """
    PROFILE_DIR contains user profiles as user_id.json
    Extract user_id numbers from filenames.
    """
    ids: List[int] = []
    try:
        if not os.path.isdir(profile_dir):
            return ids
        for name in os.listdir(profile_dir):
            if not name.lower().endswith(".json"):
                continue
            base = name[:-5]
            try:
                ids.append(int(base))
            except Exception:
                continue
    except Exception:
        return ids

    return sorted(set(ids))


def convert_ui_msgs_to_langchain(ui_messages: List[Dict[str, str]]):
    lc_messages = []
    for msg in ui_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def rebuild_graph(config: Dict[str, Any]) -> None:
    st.session_state.graph = build_graph(config)
    st.session_state.config = config
    try:
        st.session_state.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    except Exception:
        pass


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.session_state.debug_trace = []
    st.session_state.debug_trace_events = []
    st.session_state.thread_id = str(uuid.uuid4())


def init_or_reload_user(user_id: int) -> None:
    st.session_state.current_user_id = int(user_id)
    st.session_state.memory_manager = MemoryManager(user_id=int(user_id))
    st.session_state.user_profile = st.session_state.memory_manager.load_user_profile()


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

    if "debug_trace_events" not in st.session_state:
        st.session_state.debug_trace_events = []

    if "graph" not in st.session_state:
        rebuild_graph(st.session_state.config)

    if "current_user_id" not in st.session_state:
        cfg_user_id = DEFAULT_CURRENT_USER_ID
        if isinstance(st.session_state.config, dict):
            cfg_user_id = int(st.session_state.config.get("user_id", cfg_user_id))
        st.session_state.current_user_id = cfg_user_id

    if "memory_manager" not in st.session_state or "user_profile" not in st.session_state:
        init_or_reload_user(st.session_state.current_user_id)


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


# ---------------- RESPONSE EXTRACTION ----------------


def _extract_response_and_suggestion(
    node_name: str, state_update: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    response_text: Optional[str] = None
    suggestion_text: Optional[str] = None

    resp = state_update.get("response")
    if isinstance(resp, str) and resp.strip():
        response_text = resp.strip()

    sugg = state_update.get("suggestion")
    if isinstance(sugg, str) and sugg.strip():
        suggestion_text = sugg.strip()

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


# ---------------- UI ONLY NODE REPORTING ----------------


def _safe_preview(value: Any, max_len: int = 400) -> str:
    try:
        s = str(value)
    except Exception:
        return "<unprintable>"
    s = s.strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def build_ui_report(node_name: str, state_update: Dict[str, Any]) -> Dict[str, Any]:
    """
    UI-only node reporting with best-effort summaries.
    No node code changes required.
    """
    title = node_name
    level = "info"
    lines: List[str] = []
    details: Dict[str, Any] = {}

    if not isinstance(state_update, dict):
        return {
            "title": f"{node_name}: non-dict update",
            "level": "warning",
            "lines": [f"type: {type(state_update)}"],
            "details": {},
        }

    if "error" in state_update:
        level = "error"
        title = f"{node_name}: error"
        lines.append(_safe_preview(state_update.get("error")))
        details["error"] = state_update.get("error")

    if "warning" in state_update and level != "error":
        level = "warning"
        title = f"{node_name}: warning"
        lines.append(_safe_preview(state_update.get("warning")))
        details["warning"] = state_update.get("warning")

    if node_name == "INTENT":
        intent_data = state_update.get("intent_metadata")
        if intent_data:
            if isinstance(intent_data, dict):
                intent_label = intent_data.get("intent", "UNKNOWN")
                conf = intent_data.get("confidence", None)
                title = f"INTENT: {intent_label}"
                if conf is not None:
                    lines.append(f"confidence: {conf}")
                details["intent_metadata"] = intent_data
            else:
                title = "INTENT: metadata"
                details["intent_metadata"] = _safe_preview(intent_data)

    if node_name == "CLARIFICATION":
        q = state_update.get("clarification_question") or state_update.get("question") or state_update.get("clarify")
        if isinstance(q, str) and q.strip():
            title = "CLARIFICATION: asking user"
            lines.append(_safe_preview(q))
            details["clarification_question"] = q

    if node_name == "DATA_AVAILABILITY":
        missing = state_update.get("missing_fields") or state_update.get("missing_data")
        if missing:
            title = "DATA_AVAILABILITY: missing data"
            level = "warning" if level != "error" else level
            details["missing"] = missing
            if isinstance(missing, list):
                lines.append("missing: " + ", ".join([_safe_preview(x, 80) for x in missing]))
            else:
                lines.append("missing: " + _safe_preview(missing, 200))

    if node_name == "STATIC_RESPOND":
        key = state_update.get("template_key") or state_update.get("static_key")
        if key:
            title = "STATIC_RESPOND: template"
            lines.append("template: " + _safe_preview(key, 160))
            details["template"] = key

    if node_name == "PROCESS":
        title = "PROCESS: execution"
        tool = state_update.get("tool") or state_update.get("selected_tool")
        if tool:
            lines.append("tool: " + _safe_preview(tool, 160))
            details["tool"] = tool

        sql = state_update.get("sql") or state_update.get("sql_query")
        if sql:
            lines.append("sql: " + _safe_preview(sql, 240))
            details["sql"] = sql

        rag = state_update.get("sources") or state_update.get("retrieved_docs") or state_update.get("documents")
        if rag is not None:
            if isinstance(rag, list):
                lines.append(f"rag docs: {len(rag)}")
            else:
                lines.append("rag: " + _safe_preview(rag, 160))
            details["rag"] = rag

    if node_name == "SUGGESTOR":
        title = "SUGGESTOR: coaching"
        enabled = state_update.get("enabled")
        if enabled is not None:
            lines.append("enabled: " + _safe_preview(enabled, 40))

    resp_preview = state_update.get("response") or state_update.get("answer") or state_update.get("output")
    if isinstance(resp_preview, str) and resp_preview.strip():
        lines.append("response: " + _safe_preview(resp_preview, 180))

    if not lines and not details:
        keys = list(state_update.keys())
        title = f"{node_name}: update"
        lines.append("keys: " + ", ".join(keys[:12]) + ("..." if len(keys) > 12 else ""))

    return {"title": title, "level": level, "lines": lines, "details": details}


def render_ui_report(status_container, report: Dict[str, Any]) -> None:
    title = report.get("title", "update")
    level = report.get("level", "info")
    lines = report.get("lines", []) or []

    if level == "error":
        status_container.error(title)
    elif level == "warning":
        status_container.warning(title)
    else:
        status_container.write(title)

    for line in lines:
        status_container.write("- " + _safe_preview(line, 800))


# ---------------- DYNAMIC CONFIG EDITOR ----------------


def _coerce_number(value: Any, target_type: type) -> Any:
    try:
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
    except Exception:
        return value
    return value


def _render_llm_block(prefix: str, llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = dict(llm_cfg)
    model = llm_cfg.get("model", "")
    temperature = llm_cfg.get("temperature", None)
    max_tokens = llm_cfg.get("max_tokens", None)
    streaming = llm_cfg.get("streaming", None)

    model_new = st.text_input(f"{prefix}.model", value=str(model))
    llm_cfg["model"] = model_new

    if temperature is not None:
        try:
            temp_val = float(temperature)
        except Exception:
            temp_val = 0.0
        llm_cfg["temperature"] = st.slider(
            f"{prefix}.temperature", min_value=0.0, max_value=1.0, value=float(temp_val)
        )

    if max_tokens is not None:
        try:
            mt = int(max_tokens)
        except Exception:
            mt = 1024
        llm_cfg["max_tokens"] = st.number_input(f"{prefix}.max_tokens", min_value=1, value=int(mt), step=1)

    if streaming is not None:
        llm_cfg["streaming"] = st.checkbox(f"{prefix}.streaming", value=bool(streaming))

    return llm_cfg


def _render_known_numeric(path: str, key: str, value: Any) -> Any:
    """
    Render number inputs for known numeric keys.
    """
    if isinstance(value, bool):
        return st.checkbox(f"{path}.{key}", value=bool(value))

    if isinstance(value, int):
        return st.number_input(f"{path}.{key}", value=int(value), step=1)

    if isinstance(value, float):
        return st.number_input(f"{path}.{key}", value=float(value))

    return value


def render_node_config_editor(node_path: str, node_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Editor for a single runtime_nodes entry.
    Covers common patterns and falls back to JSON for unknown nested parts.
    """
    cfg = dict(node_cfg)

    if "description" in cfg:
        st.caption(str(cfg.get("description", "")))

    if "enabled" in cfg and isinstance(cfg["enabled"], bool):
        cfg["enabled"] = st.checkbox("enabled", value=bool(cfg["enabled"]))

    numeric_keys = [
        "confidence_threshold",
        "max_iterations",
        "max_history_limit",
        "max_retries",
        "retriever_k",
        "score_threshold",
        "min_relevance_score",
        "chunk_size",
        "chunk_overlap",
    ]
    for k in numeric_keys:
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = _render_known_numeric(node_path, k, cfg[k])

    llm_keys = ["llm", "llm_fast", "llm_slow", "llm_manager"]
    for lk in llm_keys:
        if lk in cfg and isinstance(cfg[lk], dict):
            st.subheader(lk)
            cfg[lk] = _render_llm_block(lk, cfg[lk])

    # Known nested blocks in rag config
    for nested in ["query_rewriter", "generate", "grade_documents", "retriever"]:
        if nested in cfg and isinstance(cfg[nested], dict):
            with st.expander(nested, expanded=False):
                nested_cfg = dict(cfg[nested])
                if "llm" in nested_cfg and isinstance(nested_cfg["llm"], dict):
                    nested_cfg["llm"] = _render_llm_block(f"{nested}.llm", nested_cfg["llm"])

                # retriever.embeddings block
                if nested == "retriever" and "embeddings" in nested_cfg and isinstance(nested_cfg["embeddings"], dict):
                    emb = dict(nested_cfg["embeddings"])
                    emb["provider"] = st.text_input("retriever.embeddings.provider", value=str(emb.get("provider", "")))
                    emb["model"] = st.text_input("retriever.embeddings.model", value=str(emb.get("model", "")))
                    emb["base_url"] = st.text_input("retriever.embeddings.base_url", value=str(emb.get("base_url", "")))
                    nested_cfg["embeddings"] = emb

                # generic numeric keys inside nested block
                for k2, v2 in list(nested_cfg.items()):
                    if k2 == "llm" or k2 == "embeddings":
                        continue
                    if isinstance(v2, (int, float, bool)):
                        nested_cfg[k2] = _render_known_numeric(f"{node_path}.{nested}", k2, v2)

                cfg[nested] = nested_cfg

    # Show refs as plain text
    for ref_key in ["sql_config_ref", "sql_validation_config_ref", "rag_config_ref", "prompt_repo", "base_url", "provider"]:
        if ref_key in cfg and isinstance(cfg[ref_key], str):
            cfg[ref_key] = st.text_input(ref_key, value=str(cfg[ref_key]))

    # Fallback JSON for anything else not covered
    uncovered: Dict[str, Any] = {}
    covered_keys = set(["description", "enabled"] + numeric_keys + llm_keys + ["query_rewriter", "generate", "grade_documents", "retriever"] + ["sql_config_ref", "sql_validation_config_ref", "rag_config_ref", "prompt_repo", "base_url", "provider"])
    for k, v in cfg.items():
        if k in covered_keys:
            continue
        if isinstance(v, (dict, list)):
            uncovered[k] = v

    if uncovered:
        with st.expander("Advanced: edit remaining JSON blocks", expanded=False):
            for k, v in uncovered.items():
                raw = st.text_area(f"{k} (JSON)", value=json.dumps(v, indent=2), height=180)
                try:
                    cfg[k] = json.loads(raw)
                except Exception:
                    st.warning(f"Invalid JSON for {k}, keeping previous value")

    return cfg


# ---------------- STREAMLIT SETUP ----------------

st.set_page_config(
    page_title="Fitbit AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("System Control")

    # Optional env debug
    with st.expander("Environment debug", expanded=False):
        st.write(f"Project Root: {project_root}")
        st.write(f"Looking for .env: {ENV_PATH}")
        st.write(f"File Found/Loaded: {is_loaded}")
        st.write(f"LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING', 'Not Set')}")
        st.write(f"LANGSMITH_API_KEY: {'[SET]' if os.environ.get('LANGSMITH_API_KEY') else '[MISSING]'}")

    st.divider()

    # User selection from PROFILE_DIR
    st.subheader("User")
    available_user_ids = safe_list_user_ids(PROFILE_DIR)
    if not available_user_ids:
        st.warning(f"No user profiles found in PROFILE_DIR: {PROFILE_DIR}")
    else:
        current_uid = int(st.session_state.current_user_id)
        if current_uid not in available_user_ids:
            current_uid = available_user_ids[0]

        selected_uid = st.selectbox(
            "Active user_id",
            options=available_user_ids,
            index=available_user_ids.index(current_uid),
        )

        if int(selected_uid) != int(st.session_state.current_user_id):
            init_or_reload_user(int(selected_uid))
            reset_conversation()
            st.rerun()

    st.divider()

    # User profile display (kept)
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

    # Metrics (kept)
    col1, col2 = st.columns(2)
    col1.metric("Turns", st.session_state.turn_count)
    col2.metric("Thread", st.session_state.thread_id[:6])

    if st.button("Reset conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()

    # Dynamic node config editor
    st.subheader("Runtime node configuration")

    config_obj = st.session_state.config if isinstance(st.session_state.config, dict) else {}
    runtime_nodes = config_obj.get("runtime_nodes", {})
    node_paths = sorted([k for k in runtime_nodes.keys() if isinstance(k, str)])

    if not node_paths:
        st.warning("No runtime_nodes found in config.")
    else:
        if "selected_node_path" not in st.session_state:
            st.session_state.selected_node_path = node_paths[0]

        selected_node = st.selectbox(
            "Select node",
            options=node_paths,
            index=node_paths.index(st.session_state.selected_node_path) if st.session_state.selected_node_path in node_paths else 0,
        )
        st.session_state.selected_node_path = selected_node

        node_cfg = get_runtime_node_cfg(st.session_state.config, selected_node)

        with st.form("node_cfg_form"):
            edited_cfg = render_node_config_editor(selected_node, node_cfg)
            apply = st.form_submit_button("Apply node config and rebuild")

        if apply:
            new_config = set_runtime_node_cfg(dict(st.session_state.config), selected_node, edited_cfg)
            save_config(new_config)
            rebuild_graph(new_config)
            st.success("Config applied.")
            st.rerun()

# ---------------- MAIN ----------------

st.title("Fitbit Health Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How are my steps looking today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    inputs = {
        "messages": convert_ui_msgs_to_langchain(st.session_state.messages),
        "user_profile": st.session_state.user_profile,
    }

    run_cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

    final_response: Optional[str] = None
    final_suggestion: Optional[str] = None
    debug_trace: List[str] = []

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        suggestion_placeholder = st.empty()
        status_container = st.status("Thinking...", expanded=True)

        try:
            for event in st.session_state.graph.stream(inputs, config=run_cfg):
                for node_name, state_update in event.items():
                    debug_trace.append(node_name)

                    report = build_ui_report(node_name, state_update if isinstance(state_update, dict) else {})
                    render_ui_report(status_container, report)

                    st.session_state.debug_trace_events.append({"node": node_name, "report": report})

                    if state_update is None or not isinstance(state_update, dict):
                        continue

                    resp_text, sugg_text = _extract_response_and_suggestion(node_name, state_update)

                    if isinstance(resp_text, str) and resp_text.strip():
                        final_response = resp_text
                        message_placeholder.markdown(final_response)

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

        if st.session_state.debug_trace_events:
            st.subheader("Node reports")
            for i, ev in enumerate(st.session_state.debug_trace_events[-80:], start=1):
                r = ev.get("report", {})
                st.write(f"{i}. {r.get('title', ev.get('node', 'NODE'))}")
                for line in r.get("lines", [])[:6]:
                    st.caption("- " + _safe_preview(line, 800))

        try:
            current_state = st.session_state.graph.get_state(run_cfg)
            st.json(current_state.values)
        except Exception:
            st.caption("State snapshot unavailable.")
