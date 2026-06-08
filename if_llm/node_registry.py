# node_registry.py — ComfyUI-specific registration: routes, NODE_CLASS_MAPPINGS.
# Imports the core IFLLM class from node_core and wires up ComfyUI integration.
# Routes are registered lazily to avoid importing ComfyUI server at module load time.

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from if_llm.node_core import IFLLM

if TYPE_CHECKING:
    from aiohttp import web as _web  # noqa: F401

logger = logging.getLogger(__name__)


# ====================================================================
# Node class mappings (ComfyUI registry)
# ====================================================================

NODE_CLASS_MAPPINGS: dict[str, type[IFLLM]] = {
    "IF_LLM": IFLLM,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "IF_LLM": "IF LLM🎨",
}


# ====================================================================
# HTTP routes — registered lazily when ComfyUI PromptServer is available
# ====================================================================


def register_routes() -> None:
    """Register IF_LLM HTTP routes with ComfyUI's PromptServer.

    Called once when ComfyUI finishes loading its node system.
    """
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        logger.error(
            "PromptServer not available. Skipping route registration for IF_LLM."
        )
        return

    if PromptServer.instance is None:
        logger.error(
            "PromptServer.instance not available. Skipping route registration for IF_LLM."
        )
        return

    @PromptServer.instance.routes.post("/IF_LLM/get_llm_models")
    async def get_llm_models_endpoint(request):
        try:
            from if_llm.model_utils import get_api_key

            data = await request.json()
            llm_provider = data.get("llm_provider")
            engine = llm_provider
            base_ip = data.get("base_ip")
            port = data.get("port")
            external_api_key = data.get("external_api_key")

            if external_api_key:
                api_key = external_api_key
            else:
                api_key_name = f"{llm_provider.upper()}_API_KEY"
                try:
                    api_key = get_api_key(api_key_name, engine)
                except ValueError:
                    api_key = None

            node = IFLLM()
            models = node.get_models(engine, base_ip, port, api_key)
            return web.json_response(models)

        except Exception as e:
            logger.error(f"Error in get_llm_models_endpoint: {str(e)}")
            return web.json_response([], status=500)

    @PromptServer.instance.routes.post("/IF_LLM/add_routes")
    async def add_routes_endpoint(request):
        return web.json_response({"status": "success"})

    @PromptServer.instance.routes.post("/IF_LLM/save_combo_settings")
    async def save_combo_settings_endpoint(request):
        try:
            from if_llm.settings_utils import (
                create_settings_from_ui,
                save_combo_settings,
            )

            data = await request.json()

            settings = create_settings_from_ui(data)

            node = IFLLM()

            saved_settings = save_combo_settings(settings, node.combo_presets_dir)

            return web.json_response(
                {
                    "status": "success",
                    "message": "Combo settings saved successfully",
                    "settings": saved_settings,
                }
            )

        except Exception as e:
            logger.error(f"Error saving combo settings: {str(e)}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)


def get_omost_function():  # type: ignore[return]
    """Lazily import omost_function only when needed.

    Returns:
        The omost_function callable, or raises ImportError if unavailable.
    """
    try:
        if "omost" not in sys.modules:
            from .omost import omost_function
        else:
            omost_function = sys.modules["omost"].omost_function
        return omost_function
    except ImportError as e:
        logger.error(f"Error importing omost_function: {e}")
        raise
