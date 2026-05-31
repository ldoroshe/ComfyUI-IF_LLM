"""Tests for image/tensor utility functions in if_llm.image_utils."""

from PIL import Image
import pytest
import torch

from if_llm.image_utils import convert_mask_to_grayscale_alpha
from if_llm.image_utils import process_mask
from if_llm.image_utils import tensor_to_base64


class TestTensorToBase64:
    def test_single_tensor(self):
        tensor = torch.rand(3, 64, 64)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)

    def test_grayscale_tensor(self):
        tensor = torch.rand(1, 64, 64)
        result = tensor_to_base64(tensor)
        assert isinstance(result, str)


class TestProcessMask:
    def test_with_tensor(self):
        image_tensor = torch.rand(1, 64, 64, 3)
        mask_tensor = torch.rand(1, 64, 64)
        result = process_mask(mask_tensor, image_tensor)
        assert result is not None

    def test_returns_correct_shape(self):
        image_tensor = torch.rand(2, 64, 64, 3)
        mask_tensor = torch.rand(1, 64, 64)
        result = process_mask(mask_tensor, image_tensor)
        assert result.shape == (2, 64, 3)


class TestConvertMaskToGrayscaleAlpha:
    def test_rgb_mask(self):
        img = Image.new("RGB", (10, 10), color="red")
        result = convert_mask_to_grayscale_alpha(img)
        assert result is not None

    def test_rgba_mask(self):
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))
        result = convert_mask_to_grayscale_alpha(img)
        assert result is not None

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            convert_mask_to_grayscale_alpha([1, 2, 3])
