# IFLLMNode.py — Backward-compatible re-export module.
# All logic has been moved to if_llm/node_core.py and if_llm/node_registry.py.
# This file preserves the original import path for external consumers.

from if_llm.node_core import IFLLM
from if_llm.node_registry import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    get_omost_function,
)

__all__ = [
    "IFLLM",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "get_omost_function",
]
