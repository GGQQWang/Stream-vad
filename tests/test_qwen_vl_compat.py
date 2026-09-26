"""Backbone selection and video-processor compatibility without model weights."""

import sys
import types

import pytest

from qwen_vl_compat import (
    language_hidden_size,
    load_vl_model,
    load_vl_processor,
    model_family,
    process_video_clips,
    resolve_pixel_budget,
)


@pytest.mark.parametrize("family,hidden", [("qwen2_vl", 3584), ("qwen3_vl", 4096)])
def test_model_config_selects_language_width(family, hidden):
    config = types.SimpleNamespace(
        model_type=family, text_config=types.SimpleNamespace(hidden_size=hidden),
    )
    assert model_family(config) == family
    assert language_hidden_size(config) == hidden


def test_rejects_unsupported_or_missing_hidden_size():
    with pytest.raises(ValueError, match="unsupported"):
        model_family(types.SimpleNamespace(model_type="other"))
    with pytest.raises(ValueError, match="hidden_size"):
        language_hidden_size(types.SimpleNamespace())


def test_loader_selects_class_from_config(monkeypatch):
    calls = []

    class Qwen2:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("qwen2", path, kwargs))
            return "model2"

    class Qwen3:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("qwen3", path, kwargs))
            return "model3"

    class AutoConfig:
        @staticmethod
        def from_pretrained(path):
            return types.SimpleNamespace(model_type="qwen3_vl" if path == "three" else "qwen2_vl")

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("processor", path, kwargs))
            return types.SimpleNamespace(video_processor=object()) if path == "three" else object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoConfig=AutoConfig,
        AutoProcessor=AutoProcessor,
        Qwen2VLForConditionalGeneration=Qwen2,
        Qwen3VLForConditionalGeneration=Qwen3,
    ))
    assert load_vl_model("two", torch_dtype="bf16") == "model2"
    assert load_vl_model("three", torch_dtype="bf16") == "model3"
    assert calls[:2] == [
        ("qwen2", "two", {"torch_dtype": "bf16"}),
        ("qwen3", "three", {"torch_dtype": "bf16"}),
    ]
    load_vl_processor("two", min_pixels=100352, max_pixels=100352)
    processor = load_vl_processor("three", min_pixels=100352, max_pixels=100352)
    assert calls[-2:] == [
        ("processor", "two", {"min_pixels": 100352, "max_pixels": 100352}),
        ("processor", "three", {}),
    ]
    assert processor.stream_vad_pixel_budget == (100352, 100352)
    assert resolve_pixel_budget("two", None, None) == (200704, 200704)
    assert resolve_pixel_budget("three", None, None) == (100352, 100352)
    assert resolve_pixel_budget("three", 50000, 120000) == (50000, 120000)


def test_qwen3_video_processor_keeps_all_sampled_frames():
    calls = []

    def video_processor(**kwargs):
        calls.append(kwargs)
        return {"pixel_values_videos": "pixels", "video_grid_thw": "grid"}

    processor = types.SimpleNamespace(
        video_processor=video_processor,
        stream_vad_pixel_budget=(100352, 100352),
    )
    clips = [types.SimpleNamespace(shape=(16, 3, 256, 256))]
    result = process_video_clips(processor, clips)
    assert result["video_grid_thw"] == "grid"
    assert calls[0]["do_sample_frames"] is False
    assert calls[0]["size"] == {
        "shortest_edge": 16 * 100352,
        "longest_edge": 16 * 100352,
    }
    with pytest.raises(ValueError, match="same sampled-frame count"):
        process_video_clips(processor, clips + [types.SimpleNamespace(shape=(8, 3, 256, 256))])


def test_qwen2_video_processor_unchanged():
    calls = []
    processor = types.SimpleNamespace(
        image_processor=lambda **kwargs: calls.append(kwargs),
    )
    clips = [object()]
    process_video_clips(processor, clips)
    assert calls == [{"images": None, "videos": clips, "return_tensors": "pt"}]
