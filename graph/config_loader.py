"""
graph/config_loader.py
Simple utility functions to load configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict

from graph.consts import BASE_DIR

DEFAULT_CONFIG_PATH = BASE_DIR / "app" / "config.json"

def load_graph_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads the full configuration dictionary from JSON.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_node_config(full_config: Dict[str, Any], module_path: str) -> Dict[str, Any]:
    """
    Helper to extract a specific node's settings from the full config dict.
    """
    runtime_nodes = full_config.get("runtime_nodes", {})

    if module_path in runtime_nodes:
        return runtime_nodes[module_path]

    print(f"Warning: Config not found for '{module_path}'")
    return {}