"""YAML dumping and combo settings management utilities."""
import os
import json
import yaml
import numpy as np


class EnhancedYAMLDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(EnhancedYAMLDumper, self).increase_indent(flow, False)


def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


EnhancedYAMLDumper.add_representer(str, str_presenter)


def numpy_int64_presenter(dumper, data):
    return dumper.represent_int(int(data))


EnhancedYAMLDumper.add_representer(np.int64, numpy_int64_presenter)


def dump_yaml(data, file_path):
    """
    Safely dumps a dictionary to a YAML file with custom formatting.
    Converts any numpy.int64 values to int to avoid YAML serialization errors.
    Uses multi-line string representation for better readability.
    """
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Convert numpy types in the entire data structure
    data = yaml.safe_load(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, Dumper=EnhancedYAMLDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True, width=1000, indent=2)


def save_combo_settings(settings_dict, combo_presets_dir):
    """Save combo settings to the AutoCombo directory."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(combo_presets_dir, exist_ok=True)
        settings_path = os.path.join(combo_presets_dir, 'combo_settings.yaml')

        with open(settings_path, 'w') as f:
            yaml.safe_dump(settings_dict, f)
        logger.info(f"Saved combo settings to {settings_path}")
        return settings_dict
    except Exception as e:
        logger.error(f"Error saving combo settings: {str(e)}")
        return None


def load_combo_settings(combo_presets_dir):
    """Load combo settings from the AutoCombo directory."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        settings_path = os.path.join(combo_presets_dir, 'combo_settings.yaml')

        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
                logger.info(f"Loaded combo settings from {settings_path}")
                return settings
        else:
            logger.warning(f"Combo settings file not found at {settings_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading combo settings: {str(e)}")
        return {}


def create_settings_from_ui(ui_settings):
    """
    Create settings.yaml from UI settings with proper type conversion.
    Handles UI values that may be boolean or string.
    """

    def convert_to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return bool(value)

    # Load profiles
    import folder_paths
    profiles_path = os.path.join(
        folder_paths.base_path,
        "custom_nodes",
        "ComfyUI-IF_LLM",
        "IF_AI",
        "presets",
        "profiles.json"
    )

    with open(profiles_path, 'r') as f:
        profiles = json.load(f)

    profile_name = ui_settings.get('profile', 'IF_PromptMKR')
    profile_content = profiles.get(profile_name, {}).get('instruction', '')

    # If 'prime_directives' is empty, use the profile content
    prime_directives = ui_settings.get('prime_directives')
    if not prime_directives or prime_directives in (None, '', 'None'):
        prime_directives = profile_content

    settings = {
        'base_ip': str(ui_settings.get('base_ip', 'localhost')),
        'port': str(ui_settings.get('port', '11434')),
        'user_prompt': str(ui_settings.get('user_prompt', 'Who helped Safiro infiltrate the Zaltar Organisation?')),
        'llm_provider': str(ui_settings.get('llm_provider', 'ollama')),
        'llm_model': str(ui_settings.get('llm_model', 'llama3.1:latest')),
        'prime_directives': prime_directives,
        'temperature': float(ui_settings.get('temperature', 0.7)),
        'max_tokens': int(ui_settings.get('max_tokens', 2048)),
        'stop_string': None if ui_settings.get('stop_string') in (None, 'None') else str(ui_settings.get('stop_string')),
        'keep_alive': convert_to_bool(ui_settings.get('keep_alive', False)),
        'clear_history': convert_to_bool(ui_settings.get('clear_history', False)),
        'history_steps': int(ui_settings.get('history_steps', 10)),
        'top_k': int(ui_settings.get('top_k', 40)),
        'top_p': float(ui_settings.get('top_p', 0.9)),
        'repeat_penalty': float(ui_settings.get('repeat_penalty', 1.2)),
        'seed': None if ui_settings.get('seed') in (None, 'None') else int(ui_settings.get('seed')),
        'external_api_key': str(ui_settings.get('external_api_key', '')),
        'random': convert_to_bool(ui_settings.get('random', False)),
        'aspect_ratio': str(ui_settings.get('aspect_ratio', '16:9')),
        'auto_combo': convert_to_bool(ui_settings.get('auto_combo', False)),
        'precision': str(ui_settings.get('precision', 'fp16')),
        'attention': str(ui_settings.get('attention', 'sdpa')),
        'batch_count': int(ui_settings.get('batch_count', 4)),
        'strategy': str(ui_settings.get('strategy', 'normal')),
        'profile': profile_name  # Include profile name
    }
    return settings


def format_response(self, response):
    """
    Format the response by adding appropriate line breaks and paragraph separations.
    """
    import re
    paragraphs = re.split(r"\n{2,}", response)

    formatted_paragraphs = []
    for para in paragraphs:
        if "```" in para:
            parts = para.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # This is a code block
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = "".join(parts)
        else:
            para = para.replace(". ", ".\n")

        formatted_paragraphs.append(para.strip())

    return "\n\n".join(formatted_paragraphs)


def print_available_models():
    """Print available models for each supported API engine"""
    from if_llm.model_utils import get_models

    # Test API key - using a dummy value since we'll mostly see fallback models
    test_api_key = "1234"

    # List of all supported engines
    engines = [
        "ollama",
        "huggingface",
        "deepseek",
        "lmstudio",
        "textgen",
        "kobold",
        "llamacpp",
        "vllm",
        "openai",
        "xai",
        "mistral",
        "groq",
        "anthropic",
        "gemini",
        "sentence_transformers",
        "transformers"
    ]

    print("\n=== Available Models by Engine ===\n")

    for engine in engines:
        print(f"\n{engine.upper()} Models:")
        print("-" * (len(engine) + 8))

        try:
            # Get models for the current engine
            models = get_models(engine, "localhost", "11434", test_api_key)

            if models:
                # Print each model with an index
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
            else:
                print("No models available or engine requires valid API key/connection")

        except Exception as e:
            print(f"Error fetching models: {str(e)}")

        print()  # Add blank line between engines


# Usage example:
if __name__ == "__main__":
    print_available_models()
