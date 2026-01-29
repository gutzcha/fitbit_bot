"""
tests/test_intent_classification.py

Refactored test harness for the Health Intent Chain.
Now includes validation for 'suggested_sources'.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Literal

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Import your actual builder
from graph.chains.intent import IntentResult, build_intent_chain
from graph.helpers import Provider, get_fast_slow_llm

load_dotenv()

# Setup rich console for better readability
console = Console()


@dataclass
class TestCase:
    query: str
    expected_category: str
    description: str


TEST_SUITE: List[TestCase] = [
    # 1. METRIC_RETRIEVAL
    TestCase(
        query="How many steps have I taken today?",
        expected_category="METRIC_RETRIEVAL",
        description="Simple stats lookup",
    ),
    TestCase(
        query="What was my sleeping heart rate last night?",
        expected_category="METRIC_RETRIEVAL",
        description="Specific metric lookup",
    ),
    # 2. CORRELATION_ANALYSIS (The "Why")
    TestCase(
        query="Why is my recovery score so low today?",
        expected_category="CORRELATION_ANALYSIS",
        description="Asking for root cause",
    ),
    TestCase(
        query="Does drinking coffee late affect my deep sleep?",
        expected_category="CORRELATION_ANALYSIS",
        description="Pattern recognition",
    ),
    # 3. COACHING_REQUEST (The "How")
    TestCase(
        query="I have 30 mins, what run should I do based on my energy?",
        expected_category="COACHING_REQUEST",
        description="Actionable advice",
    ),
    TestCase(
        query="Help me plan my sleep schedule for next week.",
        expected_category="COACHING_REQUEST",
        description="Planning assistance",
    ),
    # 4. BENCHMARK_EVALUATION (The "Is this Normal?")
    TestCase(
        query="Is a resting heart rate of 45 normal for my age?",
        expected_category="BENCHMARK_EVALUATION",
        description="Normative comparison",
    ),
    TestCase(
        query="Am I running faster than the average user?",
        expected_category="BENCHMARK_EVALUATION",
        description="Peer comparison",
    ),
    # 5. DATA / OUT OF SCOPE
    TestCase(
        query="Why is my Oura ring not syncing?",
        expected_category="DATA_AVAILABILITY",
        description="System issue",
    ),
    TestCase(
        query="Write me a python script to sort a list.",
        expected_category="OUT_OF_SCOPE",
        description="Completely unrelated",
    ),
    TestCase(
        query="Hello, what can you do?",
        expected_category="GREETING",
        description="Greeting",
    ),
    TestCase(
        query="do i dring too much coffee?",
        expected_category="BENCHMARK_EVALUATION",
        description="",
    ),
]


def run_suite(provider: Provider = "ollama"):
    """
    Runs the standard suite of tests against the chain.
    """
    console.rule(f"[bold blue]Running Test Suite with Provider: {provider}")
    fast_llm, slow_llm = get_fast_slow_llm(provider)

    # Standard Chain
    chain = build_intent_chain(fast_llm, slow_llm)

    # Expanded table to fit the new 'Sources' column
    table = Table(title="Intent Classification Results", expand=True)
    table.add_column("Query", style="cyan", no_wrap=False, min_width=25, ratio=2)
    table.add_column("Expected", style="magenta", min_width=15)
    table.add_column("Actual", style="green", min_width=15)
    table.add_column("Sources", style="yellow", min_width=20, ratio=1)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Time", justify="center", width=6)
    table.add_column("Match", justify="center", width=6)
    table.add_column("Notes", justify="right", min_width=20, ratio=1)

    for case in TEST_SUITE:
        start_ts = time.time()
        result: IntentResult = chain.invoke({"text": case.query})
        duration = time.time() - start_ts

        is_match = case.expected_category == result.intent
        match_icon = "✅" if is_match else "❌"

        # Format sources list as a clean string
        sources_str = (
            ", ".join(result.suggested_sources) if result.suggested_sources else "-"
        )

        table.add_row(
            case.query,
            case.expected_category,
            result.intent,
            sources_str,
            f"{result.confidence:.2f}",
            f"{duration:.2f}s",
            match_icon,
            result.notes,
        )

    console.print(table)


def test_slow_fallback_trigger(provider: Provider = "ollama"):
    """
    Demonstrates the Slow LLM Fallback mechanism.
    We force the fallback by setting the confidence threshold very high (0.99).
    """
    console.rule("[bold red]Testing Slow LLM Fallback Mechanism")

    fast_llm, slow_llm = get_fast_slow_llm(provider)

    # Force fallback for anything under 0.99 confidence
    high_threshold_chain = build_intent_chain(
        fast_llm,
        slow_llm,
        slow_fallback_enabled=True,
        slow_fallback_min_confidence=0.99,
    )

    ambiguous_query = (
        "I feel kinda off today, maybe it's my sleep or just stress, what do you think?"
    )

    console.print(f"[bold]Query:[/bold] '{ambiguous_query}'")
    console.print(
        f"[dim]Threshold set to 0.99. Expecting potential fallback...[/dim]\n"
    )

    result: IntentResult = high_threshold_chain.invoke({"text": ambiguous_query})

    console.print("[bold]Result Metadata:[/bold]")
    console.print(f"Intent:   [green]{result.intent}[/green]")
    console.print(f"Sources:  [yellow]{result.suggested_sources}[/yellow]")
    console.print(f"Notes:    {result.notes}")
    console.print(f"Confidence: [cyan]{result.confidence}[/cyan]")


if __name__ == "__main__":
    # 1. Run the standard suite
    run_suite(provider="ollama")

    # 2. Run the fallback test
    print("\n")
    test_slow_fallback_trigger(provider="ollama")
