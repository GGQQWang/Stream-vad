"""HIVAU-70K dataset loader for streaming VAD training.

Converts second-level event annotations to frame-level and clip-level
binary labels.  Each ``__getitem__`` returns **all non-overlapping clips**
from one video, so the downstream SSM can model cross-window temporal
dependencies.

Collate:  ``hivau_collate`` pads uneven clip counts to ``max_T`` per batch.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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


class HIVAUDataset(Dataset):
    """HIVAU-70K — per-video clip sequences.

    Each ``__getitem__`` returns:
        ``frames``: ``[T, total_sampled_frames, C, H, W]``  raw pixels.
        ``labels``: ``[T]``  binary (1 = anomaly in this clip).
        ``video_path``: str.

    Args:
        annotation_path: path to ``*_database_*.json``.
        video_root: directory containing .mp4 files.
        total_sampled_frames: frames per clip.  Default 8.
        sample_interval: stride between sampled frames (original units).
                         Default 4.
        max_windows: max consecutive clips per sample.  Long videos are
                    randomly truncated to this many clips.  Default 32.
        fps: fallback frame rate when annotation lacks ``fps``.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        video_root: str | Path,
        total_sampled_frames: int = 8,
        sample_interval: int = 4,
        max_windows: int = 32,
        fps: float = 30.0,
    ):
        super().__init__()
        self.video_root = Path(video_root)
        self.total_frames = total_sampled_frames
        self.sample_interval = sample_interval
        self.max_windows = max_windows
        self.fps = fps
        self.clip_span = total_sampled_frames * sample_interval   # original frames

        # ---- load annotations ----
        with open(annotation_path, "r") as f:
            raw = json.load(f)

        self.samples: List[dict] = []
        for video_name, meta in raw.items():
            n = meta["n_frames"]
            if n < self.clip_span:
                continue
            video_path = self.video_root / f"{video_name}.mp4"
            if not video_path.exists():
                continue

            frame_labels = _build_frame_labels(
                n, meta.get("fps", fps), meta.get("events", [])
            )
            # pre-compute clip labels
            clip_labels: List[float] = []
            clip_binary: List[int] = []
            n_clips = 0
            for start in range(0, n - self.clip_span + 1, self.clip_span):
                clip_fl = frame_labels[
                    start : start + self.clip_span : sample_interval
                ]
                clip_labels.append(float(clip_fl.mean()))            # soft ratio
                clip_binary.append(1 if clip_fl.any() else 0)       # hard for AUC
                n_clips += 1

            if n_clips == 0:
                continue

            self.samples.append({
                "video_path": str(video_path),
                "n_frames": n,
                "fps": meta.get("fps", fps),
                "clip_labels": clip_labels,           # list[float], soft ratio
                "clip_binary": clip_binary,           # list[int], hard label
                "n_clips": n_clips,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]

        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("decord is required for video reading")

        n_total = meta["n_clips"]
        max_w = min(self.max_windows, n_total)

        # Random consecutive chunk for long videos
        if n_total > max_w:
            ci_start = random.randint(0, n_total - max_w)
        else:
            ci_start = 0

        ci_end = ci_start + max_w

        vr = VideoReader(meta["video_path"], ctx=cpu(0))
        total_video_frames = len(vr)

        clips: List[torch.Tensor] = []
        for ci in range(ci_start, ci_end):
            start = ci * self.clip_span
            pts = list(range(start, start + self.clip_span, self.sample_interval))
            pts = [min(p, total_video_frames - 1) for p in pts]

            f = vr.get_batch(pts).asnumpy()                       # [T, H, W, C]
            f = torch.from_numpy(f).permute(0, 3, 1, 2)          # [T, C, H, W]
            clips.append(f)

        frames = torch.stack(clips, dim=0)                  # [max_w, Tf, C, H, W]
        chunk_ratio = meta["clip_labels"][ci_start:ci_end]
        chunk_bin = meta["clip_binary"][ci_start:ci_end]

        return {
            "video_path": meta["video_path"],
            "frames": frames,
            "labels": torch.tensor(chunk_ratio, dtype=torch.float32),   # soft ratio
            "binary": torch.tensor(chunk_bin, dtype=torch.float32),    # hard for AUC
        }


def hivau_collate(batch: List[dict]) -> dict:
    """Pad uneven clip counts across videos in a batch.

    Pads ``labels`` with ``-1`` (ignore index for BCE) and ``frames``
    with zeros (masked downstream via labels==-1).
    """
    video_paths = [b["video_path"] for b in batch]

    labels_pad = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-1.0,
    )                                                         # [B, max_T]

    B = len(batch)
    max_T = labels_pad.shape[1]
    # pad frames to max_T  (zero frames → attention mask handles them)
    frames_list: List[torch.Tensor] = []
    for b in batch:
        f = b["frames"]                          # [T, Tf, C, H, W]
        if f.shape[0] < max_T:
            pad_shape = (max_T - f.shape[0], *f.shape[1:])
            pad = torch.zeros(pad_shape, dtype=f.dtype)
            f = torch.cat([f, pad], dim=0)
        frames_list.append(f)
    frames_pad = torch.stack(frames_list, dim=0)              # [B, max_T, Tf, C, H, W]

    # attention / valid mask: True where label != -1
    valid_mask = labels_pad != -1.0                            # [B, max_T]

    return {
        "video_path": video_paths,
        "frames": frames_pad,
        "labels": labels_pad,
        "valid_mask": valid_mask,
    }
