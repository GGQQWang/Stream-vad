"""Small compatibility boundary for the Qwen2-VL and Qwen3-VL backbones."""


def model_family(config) -> str:
    family = getattr(config, "model_type", None)
    if family not in {"qwen2_vl", "qwen3_vl"}:
        raise ValueError(f"unsupported VL model_type: {family!r}")
    return family


def language_hidden_size(config) -> int:
    text_config = getattr(config, "text_config", None)
    size = getattr(text_config, "hidden_size", None)
    if size is None:
        size = getattr(config, "hidden_size", None)
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"invalid language hidden_size: {size!r}")
    return size


def resolve_pixel_budget(
    model_path: str, min_pixels: int | None, max_pixels: int | None,
) -> tuple[int, int]:
    if min_pixels is None or max_pixels is None:
        from transformers import AutoConfig

        family = model_family(AutoConfig.from_pretrained(model_path))
        default = 100352 if family == "qwen3_vl" else 200704
        min_pixels = default if min_pixels is None else min_pixels
        max_pixels = default if max_pixels is None else max_pixels
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(
            f"invalid pixel budget: min_pixels={min_pixels}, max_pixels={max_pixels}"
        )
    return min_pixels, max_pixels


def load_vl_model(model_path: str, **kwargs):
    from transformers import AutoConfig

    family = model_family(AutoConfig.from_pretrained(model_path))
    if family == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as model_class
    else:
        try:
            from transformers import Qwen3VLForConditionalGeneration as model_class
        except ImportError as exc:
            raise ImportError("Qwen3-VL requires transformers>=4.57.3") from exc
    return model_class.from_pretrained(model_path, **kwargs)


def load_vl_processor(model_path: str, *, min_pixels: int, max_pixels: int):
    from transformers import AutoConfig, AutoProcessor

    family = model_family(AutoConfig.from_pretrained(model_path))
    if family == "qwen2_vl":
        return AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels,
        )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.stream_vad_pixel_budget = (min_pixels, max_pixels)
    return processor


def process_video_clips(processor, clips):
    if hasattr(processor, "video_processor"):
        min_pixels, max_pixels = processor.stream_vad_pixel_budget
        lengths = {int(clip.shape[0]) for clip in clips}
        if len(lengths) != 1:
            raise ValueError("Qwen3-VL video clips must have the same sampled-frame count")
        n_frames = lengths.pop()
        # Qwen3 video resize budgets cover the whole video, unlike Qwen2's
        # per-frame min_pixels/max_pixels. The dataset pads each window to 16.
        return processor.video_processor(
            videos=clips,
            return_tensors="pt",
            do_sample_frames=False,
            size={
                "shortest_edge": min_pixels * n_frames,
                "longest_edge": max_pixels * n_frames,
            },
        )
    return processor.image_processor(images=None, videos=clips, return_tensors="pt")
