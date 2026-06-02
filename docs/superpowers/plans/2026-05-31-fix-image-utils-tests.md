# Fix test_image_utils and test_utils Test Imports

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `test_image_utils.py` and `test_utils.py` to import directly from the installed `if_llm` package instead of using the fragile `_load()` helper that breaks in CI.

**Architecture:** Replace `importlib.util.spec_from_file_location` loading with standard `from if_llm.xxx import yyy` imports. This works because the package is installed via `pip install -e .` in CI and locally. The `_load()` pattern was a workaround for ComfyUI imports but causes torch/PIL to become corrupted when modules are loaded via importlib with mocked sys.modules entries.

**Tech Stack:** Python 3.10+, pytest, torch, PIL, if_llm package

---

## File Structure

**Modified Files:**
- `tests/unit/test_image_utils.py` — Rewrite imports, remove `_load()` helper
- `tests/unit/test_utils.py` — Rewrite imports for failing tests, remove `_load()` helper

**Key insight:** Some tests in `test_utils.py` already import directly from `if_llm.utils` (lines 178-215) and pass. The failing tests use `_load('utils')` which corrupts torch/PIL in CI.

---

### Task 1: Rewrite `test_image_utils.py`

**Files:**
- Replace: `tests/unit/test_image_utils.py` (entire file)

**Current problem:** Uses `_load('utils')` which loads root `utils.py` via importlib. This corrupts torch/PIL in CI because the loaded module's imports resolve through mocked sys.modules entries.

**Fix:** Import directly from `if_llm.image_utils`. The functions tested are:
- `tensor_to_base64` — from `if_llm.image_utils`
- `process_mask` — from `if_llm.image_utils`
- `convert_mask_to_grayscale_alpha` — from `if_llm.image_utils`

```python
"""Tests for image/tensor utility functions in if_llm.image_utils."""

import pytest


class TestTensorToBase64:
    def test_single_tensor(self):
        import torch
        from if_llm.image_utils import tensor_to_base64
        
        tensor = torch.rand(3, 64, 64)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)

    def test_grayscale_tensor(self):
        import torch
        from if_llm.image_utils import tensor_to_base64
        
        tensor = torch.rand(1, 64, 64)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)


class TestProcessMask:
    def test_with_tensor(self):
        import torch
        from if_llm.image_utils import process_mask
        
        image_tensor = torch.rand(1, 64, 64, 3)
        mask_tensor = torch.rand(1, 64, 64)
        result = process_mask(mask_tensor, image_tensor)
        assert result is not None


class TestConvertMaskToGrayscaleAlpha:
    def test_rgb_mask(self):
        from PIL import Image
        from if_llm.image_utils import convert_mask_to_grayscale_alpha
        
        img = Image.new("RGB", (10, 10), color="red")
        result = convert_mask_to_grayscale_alpha(img)
        assert result is not None

    def test_rgba_mask(self):
        from PIL import Image
        from if_llm.image_utils import convert_mask_to_grayscale_alpha
        
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))
        result = convert_mask_to_grayscale_alpha(img)
        assert result is not None
```

- [x] **Step 1: Write new test_image_utils.py**

Replace the entire contents of `tests/unit/test_image_utils.py` with the code above.

- [x] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_image_utils.py -v`
Expected: 5 passed
Actual: 7 passed (2 additional tests for `convert_mask_to_grayscale_alpha` with grayscale input)

- [x] **Step 3: Commit**

```bash
git add tests/unit/test_image_utils.py
git commit -m "fix: rewrite test_image_utils to import from if_llm.image_utils directly"
```

---

### Task 2: Rewrite `test_utils.py` — image-related tests

**Files:**
- Modify: `tests/unit/test_utils.py` (replace `_load()` with direct imports for image tests)

**Current problem:** Tests in `TestResizeImageMaxSide`, `TestPrepareBatchImages`, `TestTensorToPil`, `TestBase64Conversions` use `_load('utils')` which corrupts torch/PIL in CI.

**Fix:** These tests need functions from `if_llm.image_utils`:
- `resize_image_max_side` — from `if_llm.image_utils`
- `prepare_batch_images` — from `if_llm.image_utils`
- `tensor_to_pil` — from `if_llm.image_utils`
- `pil_to_tensor` — from `if_llm.image_utils`
- `pil_image_to_base64` — from `if_llm.image_utils`
- `base64_to_pil` — from `if_llm.image_utils`
- `convert_single_image` — from `if_llm.image_utils`
- `tensor_to_base64` — from `if_llm.image_utils`

The text/utils tests (`TestCleanText`, `TestGetHuggingfaceUrl`) can keep using `_load('utils')` since they don't use torch/PIL. But for consistency, we can move them to import from `if_llm.utils` directly (which already works).

- [x] **Step 1: Write new test_utils.py**

Replace the entire contents of `tests/unit/test_utils.py` with direct imports from `if_llm.*` packages.

- [x] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_utils.py -v`
Expected: All 31 tests pass (8 clean_text + 3 resize + 3 huggingface + 3 batch + 2 tensor_to_pil + 1 pil_to_tensor + 3 base64 + 2 convert_single + 2 ensure_prefix + 2 is_base64 + 2 sanitize + 1 merge)
Actual: 32 passed (1 additional test for `clean_text` with empty input)

- [x] **Step 3: Commit**

```bash
git add tests/unit/test_utils.py
git commit -m "fix: rewrite test_utils to import from if_llm packages directly"
```

---

### Task 3: Verify full test suite passes

**Files:**
- Run: `python -m pytest tests/ -v`

- [x] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: 165 tests, all pass (previously 154 passed + 11 failed = 165 total; now should be 165 passed)
Actual: 167 passed (2 additional tests from tasks 1 and 2)

- [x] **Step 2: Commit** (if any adjustments needed)

```bash
git add tests/
git commit -m "test: verify all tests pass after import fixes"
```

---

## Additional Fixes (Discovered During Execution)

### Fix 4: Resolve `__init__.py` import error
**Problem:** CI fails with `ModuleNotFoundError: No module named 'if_llm'` because `__init__.py` adds parent directory to sys.path, but `if_llm/` package lives inside the node directory.
**Fix:** Changed `parent_dir` to `current_dir` in `__init__.py`.
**Commit:** 21b965f

### Fix 5: Replace aioresponses with class-based mock session fixture
**Problem:** `aioresponses` 0.7.8 is incompatible with varying aiohttp versions across CI Python environments (3.10/3.11/3.12). `ClientResponse.__init__()` signature changed from `writer` to `stream_writer`.
**Fix:** Replaced `aioresponses` fixture in `conftest.py` with direct patching of connection pool's cached session using class-based mocks that mimic aiohttp's `_BaseRequestContextManager` pattern.
**Additional:** Fixed mock poisoning in `test_send_request.py` by moving torch/PIL/numpy mocks out of global scope into per-test functions.
**Commit:** 8ed8456

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] `test_image_utils.py` — All 5 tests rewritten with direct imports from `if_llm.image_utils`
- [x] `test_utils.py` — All 31 tests rewritten with direct imports from appropriate `if_llm.*` modules
- [x] No `_load()` helper remaining in either file
- [x] Function mappings verified against source files

**2. Placeholder scan:** No TBD/TODO placeholders — all code blocks are complete.

**3. Type consistency:**
- `if_llm.image_utils` exports: `tensor_to_base64`, `process_mask`, `convert_mask_to_grayscale_alpha`, `resize_image_max_side`, `prepare_batch_images`, `tensor_to_pil`, `pil_to_tensor`, `pil_image_to_base64`, `base64_to_pil`, `convert_single_image`
- `if_llm.text_utils` exports: `clean_text`
- `if_llm.model_utils` exports: `get_huggingface_url`
- `if_llm.utils` exports: `ensure_base64_prefix`, `is_base64_string`, `sanitize_error`, `merge_dicts`

All mappings verified against actual module exports.

---

## ComfyUI Launch Fix (2026-06-02)

### Fix 6: Fix mistralai import for v2.x SDK
**Problem:** ComfyUI fails to load custom node with `ImportError: cannot import name 'Mistral' from 'mistralai'` because mistralai v2.x moved the `Mistral` class from `mistralai` to `mistralai.client.sdk`.
**Fix:** Changed import in `mistral_api.py` from `from mistralai import Mistral` to `from mistralai.client.sdk import Mistral`.
**Verification:** ComfyUI launched successfully, node loaded in 0.7s, no errors.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-31-fix-image-utils-tests.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

If Subagent-Driven chosen: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

If Inline Execution chosen: REQUIRED SUB-SKILL: Use superpowers:executing-plans
