"""Utilities for frozen visual feature caches."""

from pathlib import Path
from urllib.parse import quote

import torch


FEATURE_CACHE_VERSION = 1


def feature_cache_path(cache_root: str | Path, video_id: str) -> Path:
    safe_id = quote(video_id, safe="")
    return Path(cache_root) / f"{safe_id}.pt"


def build_feature_cache_metadata(
    *,
    video_id: str,
    n_windows: int,
    n_frames: int,
    fps: float,
    frames_per_clip: int,
    sample_interval: int,
    min_pixels: int,
    max_pixels: int,
    model_id: str,
) -> dict:
    return {
        "cache_version": FEATURE_CACHE_VERSION,
        "video_id": video_id,
        "n_windows": int(n_windows),
        "n_frames": int(n_frames),
        "fps": float(fps),
        "frames_per_clip": int(frames_per_clip),
        "sample_interval": int(sample_interval),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "model_id": str(model_id),
    }


def validate_feature_cache_metadata(
    metadata: dict,
    *,
    video_id: str,
    n_windows: int,
    n_frames: int,
    fps: float,
    frames_per_clip: int,
    sample_interval: int,
    min_pixels: int,
    max_pixels: int,
    model_id: str,
) -> None:
    expected = build_feature_cache_metadata(
        video_id=video_id,
        n_windows=n_windows,
        n_frames=n_frames,
        fps=fps,
        frames_per_clip=frames_per_clip,
        sample_interval=sample_interval,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        model_id=model_id,
    )
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(
                f"feature cache metadata mismatch for {video_id}: "
                f"{key}={actual!r}, expected {expected_value!r}"
            )


def load_feature_cache(
    cache_root: str | Path,
    *,
    video_id: str,
    n_windows: int,
    n_frames: int,
    fps: float,
    frames_per_clip: int,
    sample_interval: int,
    min_pixels: int,
    max_pixels: int,
    model_id: str,
    map_location: str | torch.device = "cpu",
) -> dict:
    path = feature_cache_path(cache_root, video_id)
    if not path.is_file():
        raise FileNotFoundError(f"feature cache not found for {video_id}: {path}")
    cache = torch.load(path, map_location=map_location)
    if "compressed_features" not in cache or "metadata" not in cache:
        raise ValueError(f"invalid feature cache file: {path}")
    validate_feature_cache_metadata(
        cache["metadata"],
        video_id=video_id,
        n_windows=n_windows,
        n_frames=n_frames,
        fps=fps,
        frames_per_clip=frames_per_clip,
        sample_interval=sample_interval,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        model_id=model_id,
    )
    features = cache["compressed_features"]
    if features.ndim != 2 or int(features.shape[0]) != int(n_windows):
        raise ValueError(
            f"feature shape mismatch for {video_id}: {tuple(features.shape)}, "
            f"expected [{n_windows}, hidden]"
        )
    return cache


def save_feature_cache_atomic(
    cache_root: str | Path,
    *,
    video_id: str,
    compressed_features: torch.Tensor,
    metadata: dict,
) -> Path:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = feature_cache_path(root, video_id)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "compressed_features": compressed_features.detach().cpu(),
        "metadata": metadata,
    }, tmp_path)
    tmp_path.replace(path)
    return path
