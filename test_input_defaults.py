#!/usr/bin/env python3
"""Self-test script to validate IF_LLM node INPUT_TYPES defaults and option lists.

Usage:
    python test_input_defaults.py

Validates:
1. Every INPUT_TYPES field has an explicit "default" value
2. Default values match expected Python types
3. Combo option lists are non-empty and contain the default value
4. Defaults are consistent between __init__, INPUT_TYPES, and process_image signature
"""

import inspect
import json
import os
import sys


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)

    from if_llm.node_core import IFLLM

    passed = 0
    failed = 0
    warnings = 0

    def check(condition, msg):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
        else:
            passed += 1
        print(f"  [{status}] {msg}")
        return condition

    def warn(condition, msg):
        nonlocal warnings
        if not condition:
            warnings += 1
            print(f"  [WARN] {msg}")
        return condition

    print("=" * 70)
    print("IF_LLM Node INPUT_TYPES Validation")
    print("=" * 70)

    # --- Test 1: Get INPUT_TYPES ---
    print("\n[1] Getting INPUT_TYPES...")
    try:
        input_types = IFLLM.INPUT_TYPES()
        check(True, "INPUT_TYPES() returned successfully")
    except Exception as e:
        print(f"  [FAIL] INPUT_TYPES() raised: {e}")
        return 1

    # --- Test 2: Check all sections exist ---
    print("\n[2] Checking required sections...")
    for section in ["required", "optional", "hidden"]:
        check(section in input_types, f"Section '{section}' exists")

    # --- Test 3: Every field has an explicit "default" ---
    print("\n[3] Checking all fields have explicit 'default' in config...")
    fields_without_default = []

    # llm_model is intentionally empty (populated dynamically by VALIDATE_INPUTS)
    allowed_no_default = {"required.llm_model"}

    for section_name, section in input_types.items():
        for field_name, field_def in section.items():
            if isinstance(field_def, tuple) and len(field_def) == 2:
                options, config = field_def
                full_name = f"{section_name}.{field_name}"
                if "default" not in config:
                    if full_name in allowed_no_default:
                        warn(
                            False, f"{full_name}: no 'default' (intentionally dynamic)"
                        )
                    else:
                        fields_without_default.append(full_name)
                        check(False, f"{full_name} has 'default'")
                else:
                    check(True, f"{full_name} has 'default' = {config['default']!r}")
            else:
                check(
                    False,
                    f"{section_name}.{field_name} is valid tuple (options, config)",
                )

    if not fields_without_default:
        print("\n  -> All fields have explicit defaults!")

    # --- Test 4: Combo lists are non-empty and contain default ---
    print("\n[4] Checking combo option lists...")
    for section_name, section in input_types.items():
        for field_name, field_def in section.items():
            if not isinstance(field_def, tuple) or len(field_def) != 2:
                continue
            options, config = field_def

            if (
                isinstance(options, list)
                and len(options) > 0
                and isinstance(options[0], str)
            ):
                default = config.get("default")
                check(
                    len(options) > 0,
                    f"{section_name}.{field_name}: option list non-empty ({len(options)} items)",
                )
                if default is not None:
                    check(
                        default in options,
                        f"{section_name}.{field_name}: default {default!r} is in option list",
                    )

    # --- Test 5: Type validation for defaults ---
    print("\n[5] Checking default value types...")
    type_map = {
        "INT": int,
        "FLOAT": float,
        "BOOLEAN": bool,
        "STRING": str,
    }

    for section_name, section in input_types.items():
        for field_name, field_def in section.items():
            if not isinstance(field_def, tuple) or len(field_def) != 2:
                continue
            options, config = field_def

            if isinstance(options, str) and options in type_map:
                default = config.get("default")
                if default is not None:
                    expected_type = type_map[options]
                    actual_type = type(default)
                    check(
                        isinstance(default, expected_type),
                        f"{section_name}.{field_name}: {options} field has {expected_type.__name__} default (got {actual_type.__name__}: {default!r})",
                    )

    # --- Test 6: __init__ defaults match INPUT_TYPES ---
    print("\n[6] Checking __init__ defaults match INPUT_TYPES...")
    node_instance = IFLLM()
    init_defaults = {
        "keep_alive": getattr(node_instance, "keep_alive", None),
        "clear_history": getattr(node_instance, "clear_history", None),
        "batch_count": getattr(node_instance, "batch_count", None),
        "seed": getattr(node_instance, "seed", None),
        "max_tokens": getattr(node_instance, "max_tokens", None),
        "history_steps": getattr(node_instance, "history_steps", None),
        "temperature": getattr(node_instance, "temperature", None),
        "top_k": getattr(node_instance, "top_k", None),
        "top_p": getattr(node_instance, "top_p", None),
        "repeat_penalty": getattr(node_instance, "repeat_penalty", None),
        "precision": getattr(node_instance, "precision", None),
        "attention": getattr(node_instance, "attention", None),
        "aspect_ratio": getattr(node_instance, "aspect_ratio", None),
    }

    input_defaults = {}
    for section_name in ["optional", "hidden"]:
        section = input_types.get(section_name, {})
        for field_name, field_def in section.items():
            if isinstance(field_def, tuple) and len(field_def) == 2:
                _, config = field_def
                if "default" in config:
                    input_defaults[field_name] = config["default"]

    for field_name, init_val in init_defaults.items():
        if field_name in input_defaults:
            input_val = input_defaults[field_name]
            check(
                init_val == input_val,
                f"{field_name}: __init__={init_val!r} matches INPUT_TYPES={input_val!r}",
            )

    # --- Test 7: process_image signature defaults match INPUT_TYPES ---
    print("\n[7] Checking process_image signature defaults match INPUT_TYPES...")
    sig = inspect.signature(IFLLM.process_image)
    for field_name, input_val in input_defaults.items():
        param = sig.parameters.get(field_name)
        if param and param.default is not inspect.Parameter.empty:
            sig_val = param.default
            # Allow "None" string in INPUT_TYPES to match None in signature (Combo fields)
            # Allow "" in INPUT_TYPES to match None in signature (forceInput STRING fields)
            compatible = (
                sig_val == input_val
                or (input_val == "None" and sig_val is None)
                or (input_val == "" and sig_val is None)
            )
            check(
                compatible,
                f"{field_name}: process_image default={sig_val!r} compatible with INPUT_TYPES={input_val!r}",
            )

    # --- Test 8: Preset files are valid JSON and non-empty ---
    print("\n[8] Checking preset files...")
    presets_dir = os.path.join(project_dir, "IF_AI", "presets")
    preset_files = [
        "profiles.json",
        "neg_prompts.json",
        "embellishments.json",
        "style_prompts.json",
        "stop_strings.json",
    ]
    for preset_file in preset_files:
        preset_path = os.path.join(presets_dir, preset_file)
        check(os.path.exists(preset_path), f"{preset_file} exists")
        if os.path.exists(preset_path):
            try:
                with open(preset_path) as f:
                    data = json.load(f)
                check(isinstance(data, dict), f"{preset_file} is a JSON object")
                check(len(data) > 0, f"{preset_file} is non-empty ({len(data)} keys)")
                check("None" in data, f"{preset_file} has 'None' key for default")
            except json.JSONDecodeError as e:
                check(False, f"{preset_file} is valid JSON: {e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed, {warnings} warnings")
    if failed == 0:
        print("STATUS: ALL CHECKS PASSED")
        return 0
    else:
        print(f"STATUS: {failed} CHECK(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
