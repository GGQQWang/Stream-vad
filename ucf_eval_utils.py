"""Lightweight UCF evaluation I/O shared by model and offline evaluators."""

from pathlib import Path

import numpy as np


def load_gt(gt_root: str | Path, video_id: str, n_frames: int) -> np.ndarray:
    path = Path(gt_root) / f"{video_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing GT file: {path}")
    gt = np.loadtxt(path, dtype=np.int64)
    gt = np.atleast_1d(gt).astype(np.int64)
    if len(gt) != int(n_frames):
        raise ValueError(f"{video_id}: GT length={len(gt)}, expected n_frames={n_frames}")
    return gt
