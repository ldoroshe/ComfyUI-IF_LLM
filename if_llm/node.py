# node.py — Backward-compatible re-export.
# IFLLMNode.py previously exported `IFLLM`. This module preserves that API.

from if_llm.node_core import IFLLM

__all__ = ["IFLLM"]
