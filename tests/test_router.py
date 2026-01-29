"""
tests/test_router.py

Integration test for the LangGraph router logic.
Verifies that:
1. Greetings -> STATIC_RESPOND
2. Ambiguous queries -> CLARIFICATION
3. Data queries -> PROCESS
"""

import os
import sys

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from graph.graph import build_graph  # Import the compiled graph

app = build_graph(
    {
        "provider": "ollama",
        "slow_fallback_enabled": True,
        "slow_fallback_min_confidence": 0.9,
    }
)
console = Console()


def run_test(query: str, expected_route: str):
    console.print(f"\n[bold blue]Testing Query:[/bold blue] '{query}'")

    # Run the graph
    # We pass an empty list of messages to start; the graph state will populate.
    inputs = {"messages": [{"role": "user", "content": query}]}

    # We want to see which nodes are visited
    # app.stream returns a generator of updates
    visited_nodes = []
    final_response = ""

    try:
        for output in app.stream(inputs):
            for node_name, state_update in output.items():
                visited_nodes.append(node_name)
                if "response" in state_update:
                    final_response = state_update["response"]
                # Print intermediate steps for debugging
                console.print(f"  [dim]Visited: {node_name}[/dim]")

        # Verification
        route_matched = expected_route in visited_nodes
        status_icon = "✅" if route_matched else "❌"

        console.print(
            f"  [bold]Route:[/bold] {status_icon} Expected [yellow]{expected_route}[/yellow], got [cyan]{visited_nodes}[/cyan]"
        )
        console.print(f"  [bold]Response:[/bold] {final_response}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def main():
    console.rule("[bold green]Running Router Integration Tests[/bold green]")

    # Case 1: Greeting (Should go to STATIC_RESPOND)
    run_test("Hello there!", expected_route="STATIC_RESPOND")

    # Case 2: Ambiguous (Should go to CLARIFICATION)
    # "How am I doing?" is listed as ambiguous in our prompts
    run_test("How am I doing?", expected_route="CLARIFICATION")

    # Case 3: Data Request (Should go to PROCESS)
    run_test("How many steps did I take today?", expected_route="PROCESS")

    # Case 4: Out of Scope (Should go to STATIC_RESPOND)
    run_test("Write me a poem about rust.", expected_route="STATIC_RESPOND")


if __name__ == "__main__":
    main()
