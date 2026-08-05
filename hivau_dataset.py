"""HIVAU-70K dataset — video chunked into fixed-size windows for training.

Each video is pre-cut into non-overlapping, contiguous chunks of at most
``max_windows`` clips.  A chunk is an independent sample; shuffling chunks
guarantees full coverage of every video in a single epoch.

Chunks carry ``valid_mask`` so the last (possibly shorter) chunk is handled
correctly in loss, metrics, and logging.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from feature_cache import load_feature_cache


def _build_frame_labels(
    n_frames: int,
    fps: float,
    events: List[List[float]],
) -> np.ndarray:
    """Convert second-level event intervals to per-frame 0/1 array."""
    labels = np.zeros(n_frames, dtype=np.uint8)
    for start_sec, end_sec in events:
        lo = max(0, int(start_sec * fps))
        hi = min(n_frames, int(end_sec * fps))
        labels[lo:hi] = 1
    return labels


def _read_video_label(meta: dict, frame_labels: np.ndarray) -> int:
    """Prefer explicit video-level labels; fall back to temporal events."""
    if "video_label" in meta:
        return int(meta["video_label"])
    if "label" in meta:
        label = meta["label"]
        if isinstance(label, str):
            return 0 if label.lower() in {"normal", "0", "negative"} else 1
        return int(label)
    return 1 if frame_labels.any() else 0


class HIVAUDataset(Dataset):
    """HIVAU-70K — per-chunk window sequences.

    Each ``__getitem__`` returns a fixed-size chunk of ``max_windows``
    consecutive clips (the last chunk may be shorter, padded with
    ``valid_mask == 0``).

    Args:
        annotation_path: ``*_database_*.json``.
        video_root: directory containing .mp4 files.
        total_sampled_frames: frames per clip.  Default 20.
        sample_interval: stride inside a clip.  Default 1 (consecutive).
        max_windows: max clips per chunk.  Default 32.
        fps: fallback frame rate.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        video_root: str | Path,
        total_sampled_frames: int = 20,
        sample_interval: int = 1,
        max_windows: int = 32,
        fps: float = 30.0,
        feature_cache_root: str | Path | None = None,
        feature_cache_model_id: str = "",
        min_pixels: int = 200704,
        max_pixels: int = 200704,
    ):
        super().__init__()
        self.video_root = Path(video_root)
        self.total_frames = total_sampled_frames
        self.sample_interval = sample_interval
        self.max_windows = max_windows
        self.fps = fps
        self.clip_span = total_sampled_frames * sample_interval
        self.feature_cache_root = Path(feature_cache_root) if feature_cache_root else None
        self.feature_cache_model_id = feature_cache_model_id
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # ---- load annotations ----
        with open(annotation_path, "r") as f:
            raw = json.load(f)

        # ---- pre-compute per-video clip labels, then chunk ----
        self.samples: List[dict] = []
        total_windows_all = 0

        for video_name, meta in raw.items():
            n = meta["n_frames"]
            if n <= 0:
                continue
            video_fps = meta.get("fps", fps)
            video_path = self.video_root / f"{video_name}.mp4"
            if self.feature_cache_root is None and not video_path.exists():
                continue

            frame_labels = _build_frame_labels(
                n, video_fps, meta.get("events", [])
            )
            video_label = _read_video_label(meta, frame_labels)
            # Per-window labels are for temporal evaluation; MIL pairing uses video_label.
            clip_soft: List[float] = []
            clip_bin: List[int] = []
            n_clips = math.ceil(n / self.clip_span)
            for ci in range(n_clips):
                start = ci * self.clip_span
                end = min(start + self.clip_span, n)
                clip_fl = frame_labels[start:end:sample_interval]
                clip_soft.append(float(clip_fl.mean()))
                clip_bin.append(1 if clip_fl.any() else 0)

            if self.feature_cache_root is not None:
                load_feature_cache(
                    self.feature_cache_root,
                    video_id=video_name,
                    n_windows=n_clips,
                    n_frames=n,
                    fps=video_fps,
                    frames_per_clip=total_sampled_frames,
                    sample_interval=sample_interval,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    model_id=feature_cache_model_id,
                    map_location="cpu",
                )

            total_windows_all += n_clips

            # ---- chunk into ≤ max_windows segments ----
            for lo in range(0, n_clips, max_windows):
                hi = min(lo + max_windows, n_clips)
                self.samples.append({
                    "video_path": str(video_path),
                    "video_id": video_name,
                    "video_label": video_label,
                    "n_frames": n,
                    "fps": video_fps,
                    "chunk_start": lo,
                    "chunk_end": hi,
                    "n_total_windows": n_clips,
                    "is_last_chunk": (hi == n_clips),
                    "clip_soft": clip_soft[lo:hi],
                    "clip_bin": clip_bin[lo:hi],
                })

        # ---- sanity checks ----
        # re-aggregate per-video
        video_window_counts: Dict[str, int] = {}
        for s in self.samples:
            vid = s["video_id"]
            w = s["chunk_end"] - s["chunk_start"]
            video_window_counts[vid] = video_window_counts.get(vid, 0) + w

        for video_name, meta in raw.items():
            if video_name not in video_window_counts:
                continue
            n_clips_ref = math.ceil(meta["n_frames"] / self.clip_span)
            assert video_window_counts[video_name] == n_clips_ref, (
                f"{video_name}: chunks cover {video_window_counts[video_name]} "
                f"windows, expected {n_clips_ref}"
            )

        self.total_raw_windows = total_windows_all
        print(
            f"HIVAUDataset: {len(self.samples)} chunks from "
            f"{len(video_window_counts)} videos, "
            f"{total_windows_all} total windows"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]
        ci_start = meta["chunk_start"]
        ci_end = meta["chunk_end"]
        n_actual = ci_end - ci_start

        if self.feature_cache_root is not None:
            cache = load_feature_cache(
                self.feature_cache_root,
                video_id=meta["video_id"],
                n_windows=meta["n_total_windows"],
                n_frames=meta["n_frames"],
                fps=meta["fps"],
                frames_per_clip=self.total_frames,
                sample_interval=self.sample_interval,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                model_id=self.feature_cache_model_id,
                map_location="cpu",
            )
            features = cache["compressed_features"][ci_start:ci_end].to(torch.float32)
            if n_actual < self.max_windows:
                pad = torch.zeros(
                    self.max_windows - n_actual, features.shape[-1],
                    dtype=features.dtype,
                )
                features = torch.cat([features, pad], dim=0)

            soft = torch.tensor(meta["clip_soft"], dtype=torch.float32)
            binary = torch.tensor(meta["clip_bin"], dtype=torch.float32)
            valid = torch.zeros(self.max_windows, dtype=torch.bool)
            valid[:n_actual] = True

            if n_actual < self.max_windows:
                soft = F.pad(soft, (0, self.max_windows - n_actual), value=-1.0)
                binary = F.pad(binary, (0, self.max_windows - n_actual), value=-1.0)

            return {
                "video_path": meta["video_path"],
                "video_id": meta["video_id"],
                "video_label": meta["video_label"],
                "chunk_start": meta["chunk_start"],
                "n_total_windows": meta["n_total_windows"],
                "is_last_chunk": meta["is_last_chunk"],
                "features": features,
                "labels": soft,
                "binary": binary,
                "valid_mask": valid,
            }

        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("decord is required for video reading")

        vr = VideoReader(meta["video_path"], ctx=cpu(0))
        total_video_frames = len(vr)
        if total_video_frames != meta["n_frames"]:
            raise ValueError(
                f"{meta['video_id']}: annotation n_frames={meta['n_frames']} "
                f"but decoded video has {total_video_frames} frames. "
                "Fix the annotation or video file before training."
            )

        clips: List[torch.Tensor] = []
        for ci in range(ci_start, ci_end):
            start = ci * self.clip_span
            end = min(start + self.clip_span, total_video_frames)
            pts = list(range(start, end, self.sample_interval))
            if len(pts) == 0:
                raise RuntimeError(f"{meta['video_id']}: empty clip at index {ci}")
            if len(pts) < self.total_frames:
                pts.extend([pts[-1]] * (self.total_frames - len(pts)))
            f = vr.get_batch(pts).asnumpy()
            f = torch.from_numpy(f).permute(0, 3, 1, 2)       # [F, C, H, W]
            clips.append(f)

        frames = torch.stack(clips, dim=0)                     # [n_actual, F, C, H, W]

        # pad to max_windows
        if n_actual < self.max_windows:
            pad = torch.zeros(
                self.max_windows - n_actual, *frames.shape[1:],
                dtype=frames.dtype,
            )
            frames = torch.cat([frames, pad], dim=0)

        soft = torch.tensor(meta["clip_soft"], dtype=torch.float32)
        binary = torch.tensor(meta["clip_bin"], dtype=torch.float32)
        valid = torch.zeros(self.max_windows, dtype=torch.bool)
        valid[:n_actual] = True

        if n_actual < self.max_windows:
            soft = F.pad(soft, (0, self.max_windows - n_actual), value=-1.0)
            binary = F.pad(binary, (0, self.max_windows - n_actual), value=-1.0)

        return {
            "video_path": meta["video_path"],
            "video_id": meta["video_id"],
            "video_label": meta["video_label"],
            "chunk_start": meta["chunk_start"],
            "n_total_windows": meta["n_total_windows"],
            "is_last_chunk": meta["is_last_chunk"],
            "frames": frames,
            "labels": soft,
            "binary": binary,
            "valid_mask": valid,
        }

def hivau_collate(batch: List[dict]) -> dict:
    """Collate — frames stay as list (different resolutions safe)."""
    out = {
        "video_path": [b["video_path"] for b in batch],
        "video_id": [b["video_id"] for b in batch],
        "video_label": [b["video_label"] for b in batch],
        "chunk_start": [b["chunk_start"] for b in batch],
        "n_total_windows": [b["n_total_windows"] for b in batch],
        "is_last_chunk": [b["is_last_chunk"] for b in batch],
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "binary": torch.stack([b["binary"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
    }
    if "features" in batch[0]:
        out["features"] = torch.stack([b["features"] for b in batch], dim=0)
    else:
        out["frames"] = [b["frames"] for b in batch]     # list of [max_w, F, C, H, W]
    return out
