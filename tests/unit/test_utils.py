"""Tests for pure utility functions in utils.py."""

import sys
sys.path.insert(0, '..')
from utils import (
    clean_text, resize_image_max_side, get_huggingface_url,
    prepare_batch_images, tensor_to_pil, pil_to_tensor,
    pil_image_to_base64, base64_to_pil, tensor_to_base64,
    convert_single_image, process_mask, convert_mask_to_grayscale_alpha
)


class TestCleanText:
    def test_removes_weights_with_ratio(self):
        result = clean_text("a (beautiful:1.3) sunset")
        assert result == "a beautiful sunset"

    def test_preserves_weights_without_ratio(self):
        result = clean_text("a (beautiful) sunset")
        assert "(beautiful)" in result

    def test_removes_author_attribution(self):
        result = clean_text("a photo by: famous artist")
        assert "by:" not in result

    def test_removes_html_tags(self):
        result = clean_text("hello <b>world</b>!")
        assert "<" not in result and ">" not in result

    def test_removes_empty_lines(self):
        result = clean_text("line one\n\n\nline two")
        assert "\n\n" not in result

    def test_preserves_intentional_line_breaks(self):
        result = clean_text("line one\nline two")
        assert "\n" in result

    def test_no_weights_flag(self):
        result = clean_text("a (beautiful:1.3) sunset", remove_weights=False)
        assert "(beautiful:1.3)" in result

    def test_empty_input(self):
        assert clean_text("") == ""


class TestResizeImageMaxSide:
    def test_image_larger_than_max(self):
        from PIL import Image
        img = Image.new("RGB", (100, 50), color="red")
        result = resize_image_max_side(img, max_size=60)
        assert max(result.size) <= 60

    def test_image_smaller_than_max(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        result = resize_image_max_side(img, max_size=100)
        assert result.size == (10, 10)

    def test_aspect_ratio_preserved(self):
        from PIL import Image
        img = Image.new("RGB", (200, 100), color="red")
        result = resize_image_max_side(img, max_size=50)
        ratio_before = 200 / 100
        ratio_after = result.size[0] / result.size[1]
        assert abs(ratio_before - ratio_after) < 0.01


class TestGetHuggingfaceUrl:
    def test_model_name_to_url(self):
        assert get_huggingface_url("gpt2") == "https://api-inference.huggingface.co/models/gpt2"

    def test_model_name_with_org(self):
        assert get_huggingface_url("meta-llama/Llama-2-7b") == "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b"

    def test_url_passthrough(self):
        url = "https://huggingface.co/models/gpt2"
        assert get_huggingface_url(url) == url


class TestPrepareBatchImages:
    def test_none_input(self):
        assert prepare_batch_images(None) == []

    def test_4d_tensor(self):
        import torch
        tensor = torch.rand(2, 64, 64, 3)
        result = prepare_batch_images(tensor)
        assert len(result) == 2

    def test_3d_tensor_hwc(self):
        import torch
        tensor = torch.rand(1, 64, 64, 3)
        result = prepare_batch_images(tensor.squeeze(0))
        assert len(result) == 1


class TestTensorToPil:
    def test_single_image_hwc(self):
        import torch
        tensor = torch.rand(1, 64, 64, 3)
        result = tensor_to_pil(tensor)
        assert result is not None

    def test_channels_first(self):
        import torch
        tensor = torch.rand(3, 64, 64)
        result = tensor_to_pil(tensor)
        assert result is not None


class TestPilToTensor:
    def test_rgb_image(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        result = pil_to_tensor(img)
        assert result is not None
        assert len(result.shape) == 3


class TestBase64Conversions:
    def test_pil_to_base64(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        result = pil_image_to_base64(img)
        assert isinstance(result, str)
        assert result.startswith("data:")

    def test_base64_to_pil(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        b64 = pil_image_to_base64(img)
        result = base64_to_pil(b64)
        assert result is not None

    def test_roundtrip(self):
        from PIL import Image
        original = Image.new("RGB", (10, 10), color="red")
        b64 = pil_image_to_base64(original)
        restored = base64_to_pil(b64)
        assert restored.size == original.size


class TestConvertSingleImage:
    def test_tensor_to_pil_format(self):
        import torch
        tensor = torch.rand(1, 64, 64, 3)
        result = convert_single_image(tensor, target_format='pil')
        assert result is not None

    def test_pil_to_tensor_format(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        result = convert_single_image(img, target_format='tensor')
        assert result is not None
