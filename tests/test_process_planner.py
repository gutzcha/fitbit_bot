"""
tests/test_process_planner.py

Refactored test harness for the PROCESS planner.
This is a live behavior runner that invokes the real planner agent and prints outputs.
No assertions, this is for manual inspection and iteration.

What it shows:
- needs_clarification and clarification_question
- selected_sources
- metrics and aggregations
- time_range
- steps
- confidence, latency, attempt count
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.table import Table

from graph.process.nodes.planner import make_planner
from graph.schemas import ConversationState, IntentMetadata

load_dotenv()

console = Console()


@dataclass
class TestCase:
    query: str
    intent: str
    suggested_sources: List[str]
    response_type: str
    description: str
    user_explicitly_asked: Optional[str] = None


TEST_SUITE: List[TestCase] = [
    TestCase(
        query="How many steps did I take today?",
        intent="METRIC_RETRIEVAL",
        suggested_sources=["USER_METRICS"],
        response_type="DATA_LOOKUP",
        description="Clear metric and time window",
    ),
    TestCase(
        query="What was my average heart rate?",
        intent="METRIC_RETRIEVAL",
        suggested_sources=["USER_METRICS"],
        response_type="DATA_LOOKUP",
        description="Should request time window for average over a period",
    ),
    TestCase(
        query="How did I do?",
        intent="COACHING_REQUEST",
        suggested_sources=["USER_METRICS", "KNOWLEDGE_BASE"],
        response_type="ACTIONABLE_ADVICE",
        description="Should request which metric and which time window",
    ),
    TestCase(
        query="Show my weight trend for the last month",
        intent="METRIC_RETRIEVAL",
        suggested_sources=["USER_METRICS"],
        response_type="TREND_ANALYSIS",
        description="Trend query with relative time window",
    ),
    TestCase(
        query="Is my heart rate normal?",
        intent="BENCHMARK_EVALUATION",
        suggested_sources=["USER_METRICS", "KNOWLEDGE_BASE"],
        response_type="BENCHMARK_INFO",
        description="Benchmark should use KB plus user metrics, needs time window",
    ),
    TestCase(
        query="How many stpes did I take toady?",
        intent="METRIC_RETRIEVAL",
        suggested_sources=["USER_METRICS"],
        response_type="DATA_LOOKUP",
        description="Typos, should still plan steps today or ask one question",
    ),
    TestCase(
        query="Explain what calories are",
        intent="OUT_OF_SCOPE",
        suggested_sources=["KNOWLEDGE_BASE"],
        response_type="HELP_MESSAGE",
        description="General explanation, KB only",
    ),
    TestCase(
        query="Compare my steps this week vs last week",
        intent="BENCHMARK_EVALUATION",
        suggested_sources=["USER_METRICS"],
        response_type="DATA_LOOKUP",
        description="Comparison with implicit windows",
    ),
    TestCase(
        query="Use this clarified question instead",
        intent="METRIC_RETRIEVAL",
        suggested_sources=["USER_METRICS"],
        response_type="DATA_LOOKUP",
        description="Conversation state override should be preferred",
        user_explicitly_asked="How many steps did I take yesterday?",
    ),
]


def _format_metrics(plan) -> str:
    metrics = getattr(plan, "metrics", []) or []
    if not metrics:
        return "-"
    parts = []
    for m in metrics:
        name = getattr(m, "name", None)
        agg = getattr(m, "aggregation", None)
        if name and agg:
            parts.append(f"{name}:{agg}")
        elif name:
            parts.append(str(name))
    return ", ".join(parts) if parts else "-"


def _format_time_range(plan) -> str:
    tr = getattr(plan, "time_range", None)
    if not tr:
        return "-"
    sd = getattr(tr, "start_date", None)
    ed = getattr(tr, "end_date", None)
    gr = getattr(tr, "granularity", None)
    if not sd and not ed:
        return "-"
    if gr:
        return f"{sd} to {ed} ({gr})"
    return f"{sd} to {ed}"


def _format_steps(plan) -> str:
    steps = getattr(plan, "steps", []) or []
    if not steps:
        return "-"
    parts = []
    for s in steps:
        step_id = getattr(s, "step_id", None)
        action = getattr(s, "action", None)
        if step_id and action:
            parts.append(f"{step_id}:{action}")
        elif action:
            parts.append(str(action))
    return ", ".join(parts) if parts else "-"


def run_suite(provider: str = "ollama"):
    console.rule(f"Running PROCESS PLANNER Suite with Provider: {provider}")

    node = make_planner(
        {
            "provider": provider,
            "planner_max_attempts": 2,
        }
    )

    table = Table(title="PROCESS Planner Results", expand=True)
    table.add_column("Query", style="cyan", no_wrap=False, min_width=28, ratio=2)
    table.add_column("Intent", style="magenta", min_width=16)
    table.add_column("Needs Clarify", style="yellow", width=12)
    table.add_column(
        "Clarification Q", style="yellow", no_wrap=False, min_width=24, ratio=2
    )
    table.add_column("Sources", style="green", min_width=18, ratio=1)
    table.add_column("Metrics", style="green", min_width=18, ratio=1)
    table.add_column("Time Range", style="green", min_width=18, ratio=1)
    table.add_column("Steps", style="white", no_wrap=False, min_width=24, ratio=2)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Time", justify="right", width=7)
    table.add_column("Attempts", justify="right", width=8)

    for case in TEST_SUITE:
        state = {
            "messages": [HumanMessage(content=case.query)],
            "intent_metadata": IntentMetadata(
                intent=case.intent,
                confidence=0.9,
                suggested_sources=case.suggested_sources,
                response_type=case.response_type,
            ),
            "conversation_state": ConversationState(
                current_topic=None,
                mentioned_metrics=[],
                prior_intent=None,
                user_explicitly_asked=case.user_explicitly_asked,
            ),
        }

        start_ts = time.time()
        update = node(state)
        duration = time.time() - start_ts

        plan = update.get("process_plan")

        needs_clarify = "Y" if update.get("needs_clarification") else "N"

        clarification_q = "-"
        if plan is not None:
            cq = getattr(plan, "clarification_question", None)
            if cq:
                clarification_q = str(cq)

        sources_str = "-"
        if plan is not None:
            sources = getattr(plan, "selected_sources", []) or []
            sources_str = ", ".join(sources) if sources else "-"

        conf_str = "-"
        if plan is not None:
            conf = getattr(plan, "confidence", None)
            if conf is not None:
                try:
                    conf_str = f"{float(conf):.2f}"
                except Exception:
                    conf_str = str(conf)

        attempts = update.get("planner_attempts", "-")

        table.add_row(
            case.query,
            case.intent,
            needs_clarify,
            clarification_q,
            sources_str,
            _format_metrics(plan) if plan is not None else "-",
            _format_time_range(plan) if plan is not None else "-",
            _format_steps(plan) if plan is not None else "-",
            conf_str,
            f"{duration:.2f}s",
            str(attempts),
        )

    console.print(table)


if __name__ == "__main__":
    run_suite(provider="anthropic")
