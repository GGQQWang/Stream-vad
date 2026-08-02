"""HIVAU-70K dataset loader for streaming VAD training.

Converts second-level event annotations to frame-level and clip-level
binary labels on the fly.  Videos are read as .mp4 via decord.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _build_frame_labels(
    n_frames: int,
    fps: float,
    events: List[List[float]],
) -> np.ndarray:
    """Convert second-level event intervals to a per-frame 0/1 array.

    Args:
        n_frames: total frame count of the video.
        fps: frame rate.
        events: list of ``[start_sec, end_sec]`` intervals.

    Returns:
        ``labels``: ``[n_frames]``  uint8 array, 1 = anomaly, 0 = normal.
    """
    labels = np.zeros(n_frames, dtype=np.uint8)
    for start_sec, end_sec in events:
        lo = int(start_sec * fps)
        hi = int(end_sec * fps)
        lo = max(0, lo)
        hi = min(n_frames, hi)
        labels[lo:hi] = 1
    return labels


class HIVAUDataset(Dataset):
    """HIVAU-70K dataset.

    Reads raw_annotations JSON, generates frame-level labels from
    second-level events, and returns uniformly sampled 8-frame clips
    with clip-level anomaly labels.

    Args:
        annotation_path: path to one of the ``*_database_*.json`` files.
        video_root: root directory containing the .mp4 files.
        total_sampled_frames: frames per clip.  Default 8.
        sample_interval: stride between sampled frames (in original
                        frame units).  Default 4 (matches UBnormal).
        llm_sample_frames: number of frames sent to LLM (conditional
                          frames).  Default 8 (all frames).
        train: if True, shuffle clips; if False, iterate in order.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        video_root: str | Path,
        total_sampled_frames: int = 8,
        sample_interval: int = 4,
        llm_sample_frames: int = 8,
        fps: float = 30.0,
        train: bool = True,
    ):
        super().__init__()
        self.video_root = Path(video_root)
        self.total_frames = total_sampled_frames
        self.sample_interval = sample_interval
        self.llm_frames = llm_sample_frames
        self.fps = fps
        self.train = train
        self.clip_span = total_sampled_frames * sample_interval  # original frames

        # ---- load annotations ----
        with open(annotation_path, "r") as f:
            raw = json.load(f)

        self.samples: List[dict] = []
        for video_name, meta in raw.items():
            # skip videos with no events (normal-only) or missing fps
            if not meta.get("events"):
                continue
            video_path = self.video_root / f"{video_name}.mp4"
            if not video_path.exists():
                continue

            frame_labels = _build_frame_labels(
                meta["n_frames"],
                meta.get("fps", fps),
                meta["events"],
            )
            meta["_frame_labels"] = frame_labels
            meta["_video_path"] = str(video_path)
            self.samples.append(meta)

        # ---- pre-compute clip indices for each video ----
        self.clips: List[Tuple[int, int, int]] = []  # (sample_idx, start_frame, label)
        for s_idx, meta in enumerate(self.samples):
            n = meta["n_frames"]
            fl = meta["_frame_labels"]
            # generate non-overlapping clips
            for start in range(0, n - self.clip_span + 1, self.clip_span):
                clip_fl = fl[start: start + self.clip_span: sample_interval]
                label = 1 if clip_fl.any() else 0
                self.clips.append((s_idx, start, label))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        s_idx, start, label = self.clips[idx]
        meta = self.samples[s_idx]

        # decord lazy import (not a hard dependency for the compression pkg)
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("decord is required for video reading")

        vr = VideoReader(meta["_video_path"], ctx=cpu(0))
        frame_pts = list(range(start, start + self.clip_span, self.sample_interval))
        frame_pts = [min(p, len(vr) - 1) for p in frame_pts]

        frames = vr.get_batch(frame_pts).asnumpy()          # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

        # frame-level labels for the sampled frames
        fl = meta["_frame_labels"]
        gt = torch.from_numpy(
            fl[start: start + self.clip_span: self.sample_interval].copy()
        ).long()

        return {
            "video_path": meta["_video_path"],
            "frames": frames,              # [T, C, H, W]  raw pixels
            "clip_label": torch.tensor(label).long(),  # scalar 0/1
            "frame_labels": gt,            # [T]  0/1 per sampled frame
        }
