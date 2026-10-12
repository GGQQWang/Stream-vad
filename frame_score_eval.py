"""Shared window-to-frame mapping for Stage-1 and zero-shot inference."""

from __future__ import annotations

import numpy as np


def metadata_without_test_annotations(n_frames: int, fps: float) -> dict:
    """Keep test-set indexing independent of HIVAU labels and event templates."""
    return {
        "n_frames": n_frames, "fps": fps, "video_label": 0,
        "label": [], "events": [], "events_summary_split": [],
        "clips": [], "clips_caption": [],
    }


def frame_scores_from_windows(rows: list[dict], n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standard = np.zeros(n_frames, dtype=np.float32)
    causal = np.full(n_frames, np.nan, dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)
    for row in rows:
        start = max(0, int(row["start_frame"]))
        end = min(n_frames, int(row["end_frame"]))
        if end > start:
            standard[start:end] = float(row["score_prob"])
    for index, row in enumerate(rows):
        start = min(n_frames, int(row["end_frame"]))
        next_end = n_frames if index + 1 == len(rows) else min(n_frames, int(rows[index + 1]["end_frame"]))
        if next_end > start:
            causal[start:next_end] = float(row["score_prob"])
            valid[start:next_end] = True
    return standard, causal, valid


def auc_ap(scores: np.ndarray, gt: np.ndarray) -> tuple[float | None, float | None]:
    if np.unique(gt).size < 2:
        return None, None
    from sklearn.metrics import average_precision_score, roc_auc_score

    return float(roc_auc_score(gt, scores)), float(average_precision_score(gt, scores))
