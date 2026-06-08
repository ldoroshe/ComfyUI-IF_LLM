import os
import sys

# Add this directory to path so if_llm package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import folder_paths (used for model path handling)
# models_dir is imported by IFLLM class in node_core.py, not here

from .IFDisplayTextWildcardNode import IFDisplayTextWildcard
from .IFLLMDisplayOmniNode import IFDisplayOmni
from .IFLLMDisplayTextNode import IFDisplayText
from .IFLLMJoinTextNode import IFJoinText
from .IFLLMLoadImagesNodeS import IFLoadImagess

# Then import your other modules
from .IFLLMNode import IFLLM
from .IFLLMSaveTextNode import IFSaveText
from .IFLLMTextTyperNode import IFTextTyper
from .ListModelsNode import ListModelsNode

try:
    from .send_request import *
except Exception as e:
    print(f"Warning: Could not import send_request at startup: {e}")

# Register IF_LLM HTTP routes with ComfyUI server (after all imports resolved)
try:
    from if_llm.node_registry import register_routes

    register_routes()
except Exception as e:
    print(f"Warning: Could not register IF_LLM routes: {e}")


"""# Unified omost import handling
try:
    if "omost" not in sys.modules:  # Check if already imported
        try:
            from .omost import omost_function  # Try relative import first
            print("Successfully imported omost_function from current directory")
        except ImportError:
            # If relative import fails, try from parent directory
            parent_dir = os.path.dirname(current_dir)
            parent_dir_name = os.path.basename(parent_dir)
            if parent_dir_name == 'ComfyUI-IF_LLM':
                sys.path.insert(0, parent_dir)
                from omost import omost_function
                print(f"Successfully imported omost_function from {parent_dir}/omost.py")
    else:
        omost_function = sys.modules["omost"].omost_function
        print("Using already imported omost_function")
except ImportError as e:
    print(f"Error importing omost_function: {e}")
    print(f"Current sys.path: {sys.path}")
    raise"""


class OmniType(str):
    """A special string type that acts as a wildcard for universal input/output.
    It always evaluates as equal in comparisons."""

    def __ne__(self, __value: object) -> bool:
        return False


OMNI = OmniType("*")

NODE_CLASS_MAPPINGS = {
    "IF_LLM": IFLLM,
    "IF_LLM_SaveText": IFSaveText,
    "IF_LLM_DisplayText": IFDisplayText,
    "IF_LLM_DisplayTextWildcard": IFDisplayTextWildcard,
    "IF_LLM_DisplayOmni": IFDisplayOmni,
    "IF_LLM_TextTyper": IFTextTyper,
    "IF_LLM_JoinText": IFJoinText,
    "IF_LLM_LoadImagesS": IFLoadImagess,
    "IF_LLM_ListModels": ListModelsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_LLM": "IF LLM🎨",
    "IF_LLM_SaveText": "IF Save Text📝",
    "IF_LLM_DisplayText": "IF Display Text📟",
    "IF_LLM_DisplayTextWildcard": "IF Display Text Wildcard📟",
    "IF_LLM_DisplayOmni": "IF Display Omni🔍",
    "IF_LLM_TextTyper": "IF Text Typer✍️",
    "IF_LLM_JoinText": "IF Join Text 📝",
    "IF_LLM_LoadImagesS": "IF LLM Load Images S 🖼️",
    "IF_LLM_ListModels": "IF LLM List Models 📚",
}

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "omost_function",
]
