"""Utilities for frozen visual feature caches.

v3 caches additionally store the per-window spatial compression output
(``spatial_features`` + ``spatial_mask``) for the world-model branch;
``compressed_features`` stays byte-identical to v2 for the VAD path.
v2 caches remain readable when the world model is disabled.
"""

from pathlib import Path
from urllib.parse import quote

import torch


FEATURE_CACHE_VERSION = 3


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
        "source_fps": float(fps),
        "sampled_fps": float(fps) / float(sample_interval),
        "frames_per_clip": int(frames_per_clip),
        "sample_interval": int(sample_interval),
        "window_span_frames": int(frames_per_clip) * int(sample_interval),
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
        if key == "cache_version":
            # v2 caches stay readable for the VAD-only path; strict
            # version equality is checked separately
            continue
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
    require_spatial: bool = False,
) -> dict:
    path = feature_cache_path(cache_root, video_id)
    if not path.is_file():
        raise FileNotFoundError(f"feature cache not found for {video_id}: {path}")
    cache = torch.load(path, map_location=map_location, weights_only=True)
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
    if require_spatial:
        for key in ("spatial_features", "spatial_mask"):
            if key not in cache:
                raise ValueError(
                    f"world model requires feature cache v3 with "
                    f"spatial_features for {video_id}; "
                    "rerun precompute_visual_features.py"
                )
        sf = cache["spatial_features"]
        if sf.ndim != 3 or int(sf.shape[0]) != int(n_windows):
            raise ValueError(
                f"spatial_features shape mismatch for {video_id}: "
                f"{tuple(sf.shape)}, expected [{n_windows}, R_max, hidden]"
            )
    return cache


def save_feature_cache_atomic(
    cache_root: str | Path,
    *,
    video_id: str,
    compressed_features: torch.Tensor,
    metadata: dict,
    spatial_features: torch.Tensor | None = None,
    spatial_mask: torch.Tensor | None = None,
) -> Path:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = feature_cache_path(root, video_id)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "compressed_features": compressed_features.detach().cpu(),
        "metadata": metadata,
    }
    if spatial_features is not None:
        payload["spatial_features"] = spatial_features.detach().cpu()
        payload["spatial_mask"] = spatial_mask.detach().cpu()
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path
