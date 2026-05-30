"""Tests for image/tensor utility functions in utils.py."""

import os
import importlib.util

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def _load(name):
    """Load a module directly from file."""
    filepath = os.path.join(_PROJECT_ROOT, f'{name}.py')
    spec = importlib.util.spec_from_file_location(f'if_llm_{name}', filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_utils = _load('utils')
tensor_to_base64 = _utils.tensor_to_base64
process_mask = _utils.process_mask
convert_mask_to_grayscale_alpha = _utils.convert_mask_to_grayscale_alpha


class TestTensorToBase64:
    def test_single_tensor(self):
        import torch
        tensor = torch.rand(3, 64, 64)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)

    def test_grayscale_tensor(self):
        import torch
        tensor = torch.rand(1, 64, 64)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)


class TestProcessMask:
    def test_with_tensor(self):
        import torch
        image_tensor = torch.rand(1, 64, 64, 3)
        mask_tensor = torch.rand(1, 64, 64)
        result = process_mask(mask_tensor, image_tensor)
        assert result is not None


class TestConvertMaskToGrayscaleAlpha:
    def test_rgb_mask(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        result = convert_mask_to_grayscale_alpha(img)
        assert result is not None

    def test_rgba_mask(self):
        from PIL import Image
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))
        result = convert_mask_to_grayscale_alpha(img)
        assert result is not None
