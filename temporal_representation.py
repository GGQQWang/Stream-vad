"""Offline temporal representation metrics over final-layer SCORE hidden states."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from stage_ap import _validate_window_timeline
from ucf_eval_utils import load_gt


CACHE_SCHEMA_VERSION = 1
DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
EPS = 1e-12
PURE_MIXED = -1


def l2_normalize(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"representations must be rank-2, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(eps))


def _segment_ids(frame_gt: np.ndarray) -> np.ndarray:
    labels = (np.asarray(frame_gt).reshape(-1) != 0).astype(np.int8)
    if len(labels) == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.r_[True, labels[1:] != labels[:-1]]
    return np.cumsum(starts, dtype=np.int64) - 1


def classify_windows(
    frame_gt: np.ndarray,
    window_index: np.ndarray,
    start_frame: np.ndarray,
    valid_end_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pure binary label and contiguous GT segment ID per window.

    Mixed windows receive ``-1`` for both values. Timeline validation is the
    exact validator used by the existing Stage-AP path.
    """
    gt = (np.asarray(frame_gt).reshape(-1) != 0).astype(np.int8)
    indices = np.asarray(window_index, dtype=np.int64)
    starts = np.asarray(start_frame, dtype=np.int64)
    ends = np.asarray(valid_end_frame, dtype=np.int64)
    if not (indices.shape == starts.shape == ends.shape) or indices.ndim != 1:
        raise ValueError("window index/start/end arrays must be matching rank-1 arrays")
    rows = [
        {"window_index": int(index), "start_frame": int(start), "end_frame": int(end)}
        for index, start, end in zip(indices, starts, ends)
    ]
    _validate_window_timeline(gt, rows)
    frame_segments = _segment_ids(gt)
    labels = np.full(len(indices), PURE_MIXED, dtype=np.int8)
    segments = np.full(len(indices), PURE_MIXED, dtype=np.int64)
    for position, (start, end) in enumerate(zip(starts, ends)):
        window_labels = gt[start:end]
        window_segments = frame_segments[start:end]
        if (
            len(window_labels) > 0
            and np.all(window_labels == window_labels[0])
            and np.all(window_segments == window_segments[0])
        ):
            labels[position] = window_labels[0]
            segments[position] = window_segments[0]
    return labels, segments


def k_block_representations(
    normalized_hidden: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stride-1, same-segment normal/anomaly k-window representations."""
    hidden = np.asarray(normalized_hidden, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    segments = np.asarray(segment_ids, dtype=np.int64)
    if k < 1:
        raise ValueError("k must be positive")
    if hidden.ndim != 2 or len(hidden) != len(labels) or len(labels) != len(segments):
        raise ValueError("hidden, labels, and segment_ids must have matching window dimension")
    empty = np.empty((0, hidden.shape[1]), dtype=np.float32)
    if len(hidden) < k:
        return empty, empty.copy()
    segment_windows = np.lib.stride_tricks.sliding_window_view(segments, k)
    same_segment = (
        (segment_windows[:, 0] >= 0)
        & np.all(segment_windows == segment_windows[:, :1], axis=1)
    )
    prefix = np.concatenate([
        np.zeros((1, hidden.shape[1]), dtype=np.float32),
        np.cumsum(hidden, axis=0, dtype=np.float32),
    ], axis=0)
    means = (prefix[k:] - prefix[:-k]) / np.float32(k)
    means = l2_normalize(means)
    block_labels = labels[:len(means)]
    normal = means[same_segment & (block_labels == 0)]
    anomaly = means[same_segment & (block_labels == 1)]
    return normal, anomaly


def fisher_statistics(
    normal: np.ndarray,
    anomaly: np.ndarray,
    eps: float = EPS,
) -> dict[str, float]:
    normal = np.asarray(normal, dtype=np.float32)
    anomaly = np.asarray(anomaly, dtype=np.float32)
    if normal.ndim != 2 or anomaly.ndim != 2 or normal.shape[1:] != anomaly.shape[1:]:
        raise ValueError("normal and anomaly samples must be compatible rank-2 arrays")
    if len(normal) < 1 or len(anomaly) < 1:
        raise ValueError("Fisher statistics require both classes")
    normal_mean = normal.mean(axis=0)
    anomaly_mean = anomaly.mean(axis=0)
    normal_scatter = float(np.mean(np.sum((normal - normal_mean) ** 2, axis=1)))
    anomaly_scatter = float(np.mean(np.sum((anomaly - anomaly_mean) ** 2, axis=1)))
    between_scatter = float(np.sum((normal_mean - anomaly_mean) ** 2))
    fdr = between_scatter / (normal_scatter + anomaly_scatter + eps)
    return {
        "fdr": float(fdr),
        "normal_scatter": normal_scatter,
        "anomaly_scatter": anomaly_scatter,
        "between_scatter": between_scatter,
    }


def variation_groups(
    normalized_hidden: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    start_frame: np.ndarray,
    valid_end_frame: np.ndarray,
    frame_gt: np.ndarray,
    k: int,
) -> dict:
    """Return per-segment/event/boundary distance groups for one video."""
    hidden = np.asarray(normalized_hidden, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    segments = np.asarray(segment_ids, dtype=np.int64)
    starts = np.asarray(start_frame, dtype=np.int64)
    ends = np.asarray(valid_end_frame, dtype=np.int64)
    gt = (np.asarray(frame_gt).reshape(-1) != 0).astype(np.int8)
    if k < 1:
        raise ValueError("k must be positive")
    result = {"normal": {}, "anomaly": {}, "boundary": {}}
    if len(hidden) <= k:
        return result

    distances = 1.0 - np.sum(hidden[:-k] * hidden[k:], axis=1)
    distances = np.clip(distances, 0.0, 2.0).astype(np.float32)
    segment_windows = np.lib.stride_tricks.sliding_window_view(segments, k + 1)
    same_segment = (
        (segment_windows[:, 0] >= 0)
        & np.all(segment_windows == segment_windows[:, :1], axis=1)
    )
    for label, name in ((0, "normal"), (1, "anomaly")):
        mask = same_segment & (labels[:-k] == label)
        for segment_id in np.unique(segments[:-k][mask]):
            positions = mask & (segments[:-k] == segment_id)
            result[name][int(segment_id)] = distances[positions]

    endpoint_mask = (
        (segments[:-k] >= 0)
        & (segments[k:] >= 0)
        & (labels[:-k] != labels[k:])
    )
    boundary_at = np.zeros(len(gt), dtype=np.int64)
    boundary_at[1:] = gt[1:] != gt[:-1]
    boundary_prefix = np.concatenate([[0], np.cumsum(boundary_at, dtype=np.int64)])
    span_starts = starts[:-k]
    span_ends = ends[k:]
    transition_count = boundary_prefix[span_ends] - boundary_prefix[span_starts + 1]
    boundary_mask = endpoint_mask & (transition_count == 1)
    boundary_frames = np.flatnonzero(boundary_at)
    for position in np.flatnonzero(boundary_mask):
        candidate = int(np.searchsorted(boundary_frames, span_starts[position], side="right"))
        boundary_frame = int(boundary_frames[candidate])
        if boundary_frame >= span_ends[position]:
            raise AssertionError("identified boundary is outside the pair interval")
        direction = f"{int(labels[position])}->{int(labels[position + k])}"
        key = (boundary_frame, direction)
        result["boundary"].setdefault(key, []).append(float(distances[position]))
    for key, values in list(result["boundary"].items()):
        result["boundary"][key] = np.asarray(values, dtype=np.float32)
    return result


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(_json_safe(value), handle, indent=2)
    os.replace(temporary, path)


def save_repr_video(
    path: str | Path,
    *,
    method: str,
    video_id: str,
    n_frames: int,
    fps: float,
    window_index: np.ndarray,
    start_frame: np.ndarray,
    valid_end_frame: np.ndarray,
    score_hidden: np.ndarray,
    storage_dtype: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = np.asarray(score_hidden, dtype=np.float32)
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    hidden = hidden.astype(np.float16 if storage_dtype == "float16" else np.float32)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
            method=np.asarray(method),
            video_id=np.asarray(video_id),
            n_frames=np.asarray(n_frames, dtype=np.int64),
            fps=np.asarray(fps, dtype=np.float64),
            window_index=np.asarray(window_index, dtype=np.int64),
            start_frame=np.asarray(start_frame, dtype=np.int64),
            valid_end_frame=np.asarray(valid_end_frame, dtype=np.int64),
            score_hidden=hidden,
        )
    os.replace(temporary, path)


def load_repr_video(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version", "method", "video_id", "n_frames", "fps", "window_index",
            "start_frame", "valid_end_frame", "score_hidden",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing representation fields {sorted(missing)}")
        out = {key: data[key].copy() for key in required}
    if int(out["schema_version"]) != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported cache schema {int(out['schema_version'])}")
    hidden = np.asarray(out["score_hidden"])
    count = len(np.asarray(out["window_index"]))
    if hidden.ndim != 2 or hidden.shape[0] != count or hidden.shape[1] < 1:
        raise ValueError(f"{path}: invalid score_hidden shape {hidden.shape}")
    if not np.isfinite(hidden).all():
        raise ValueError(f"{path}: score_hidden contains non-finite values")
    for key in ("start_frame", "valid_end_frame"):
        if np.asarray(out[key]).shape != (count,):
            raise ValueError(f"{path}: {key} must have shape ({count},)")
    return {
        "method": str(out["method"].item()),
        "video_id": str(out["video_id"].item()),
        "n_frames": int(out["n_frames"]),
        "fps": float(out["fps"]),
        "window_index": np.asarray(out["window_index"], dtype=np.int64),
        "start_frame": np.asarray(out["start_frame"], dtype=np.int64),
        "valid_end_frame": np.asarray(out["valid_end_frame"], dtype=np.int64),
        "score_hidden": hidden.astype(np.float32),
        "storage_dtype": str(hidden.dtype),
    }


def evaluate_representation_cache(
    cache_dir: str | Path,
    gt_root: str | Path,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> list[dict]:
    cache_dir = Path(cache_dir)
    with open(cache_dir / "manifest.json") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{cache_dir}: unsupported manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError(f"{cache_dir}: representation cache is not complete")
    method = str(manifest["method"])
    entries = sorted(manifest["videos"], key=lambda item: item["video_id"])
    k_values = tuple(int(k) for k in k_values)
    if not k_values or any(k < 1 for k in k_values) or len(set(k_values)) != len(k_values):
        raise ValueError("k_values must be unique positive integers")

    accumulators = {
        k: {
            "fdr": [], "normal_scatter": [], "anomaly_scatter": [],
            "between_scatter": [], "normal_blocks": 0, "anomaly_blocks": 0,
            "normal_blocks_used": 0, "anomaly_blocks_used": 0,
            "fdr_skips": Counter(), "fdr_skipped_videos": [], "normal_segment_means": [],
            "anomaly_segment_means": [], "boundary_means": [],
            "normal_pair_count": 0, "anomaly_pair_count": 0,
            "boundary_pair_count": 0, "normal_distance_sum": 0.0,
            "anomaly_distance_sum": 0.0, "boundary_distance_sum": 0.0,
        }
        for k in k_values
    }

    for entry in entries:
        video = load_repr_video(cache_dir / entry["file"])
        if video["method"] != method:
            raise ValueError(f"cache method mismatch: {video['method']!r} vs {method!r}")
        if video["video_id"] != entry["video_id"]:
            raise ValueError(f"cache entry/video mismatch: {entry['video_id']} vs {video['video_id']}")
        gt = load_gt(gt_root, video["video_id"], video["n_frames"])
        labels, segments = classify_windows(
            gt,
            video["window_index"],
            video["start_frame"],
            video["valid_end_frame"],
        )
        hidden = l2_normalize(video["score_hidden"])
        for k in k_values:
            accumulator = accumulators[k]
            normal, anomaly = k_block_representations(hidden, labels, segments, k)
            accumulator["normal_blocks"] += len(normal)
            accumulator["anomaly_blocks"] += len(anomaly)
            if len(normal) >= 2 and len(anomaly) >= 2:
                stats = fisher_statistics(normal, anomaly)
                for key in ("fdr", "normal_scatter", "anomaly_scatter", "between_scatter"):
                    accumulator[key].append(stats[key])
                accumulator["normal_blocks_used"] += len(normal)
                accumulator["anomaly_blocks_used"] += len(anomaly)
            else:
                if len(normal) < 2 and len(anomaly) < 2:
                    reason = "insufficient_both_classes"
                elif len(normal) < 2:
                    reason = "insufficient_normal_blocks"
                else:
                    reason = "insufficient_anomaly_blocks"
                accumulator["fdr_skips"][reason] += 1
                accumulator["fdr_skipped_videos"].append({
                    "video_id": video["video_id"],
                    "reason": reason,
                    "normal_blocks": len(normal),
                    "anomaly_blocks": len(anomaly),
                })

            groups = variation_groups(
                hidden,
                labels,
                segments,
                video["start_frame"],
                video["valid_end_frame"],
                gt,
                k,
            )
            for group_name, macro_key, count_key, sum_key in (
                ("normal", "normal_segment_means", "normal_pair_count", "normal_distance_sum"),
                ("anomaly", "anomaly_segment_means", "anomaly_pair_count", "anomaly_distance_sum"),
                ("boundary", "boundary_means", "boundary_pair_count", "boundary_distance_sum"),
            ):
                for distances in groups[group_name].values():
                    accumulator[macro_key].append(float(np.mean(distances)))
                    accumulator[count_key] += len(distances)
                    accumulator[sum_key] += float(np.sum(distances, dtype=np.float64))

    rows = []
    for k in k_values:
        acc = accumulators[k]
        rows.append({
            "method": method,
            "k": k,
            "fdr": _safe_mean(acc["fdr"]),
            "normal_scatter": _safe_mean(acc["normal_scatter"]),
            "anomaly_scatter": _safe_mean(acc["anomaly_scatter"]),
            "between_scatter": _safe_mean(acc["between_scatter"]),
            "v_normal": _safe_mean(acc["normal_segment_means"]),
            "v_anomaly": _safe_mean(acc["anomaly_segment_means"]),
            "v_boundary": _safe_mean(acc["boundary_means"]),
            "v_normal_micro": (
                acc["normal_distance_sum"] / acc["normal_pair_count"]
                if acc["normal_pair_count"] else math.nan
            ),
            "v_anomaly_micro": (
                acc["anomaly_distance_sum"] / acc["anomaly_pair_count"]
                if acc["anomaly_pair_count"] else math.nan
            ),
            "v_boundary_micro": (
                acc["boundary_distance_sum"] / acc["boundary_pair_count"]
                if acc["boundary_pair_count"] else math.nan
            ),
            "fdr_valid_videos": len(acc["fdr"]),
            "fdr_skip_reasons": dict(sorted(acc["fdr_skips"].items())),
            "fdr_skipped_videos": acc["fdr_skipped_videos"],
            "normal_block_count": acc["normal_blocks"],
            "anomaly_block_count": acc["anomaly_blocks"],
            "fdr_normal_block_count_used": acc["normal_blocks_used"],
            "fdr_anomaly_block_count_used": acc["anomaly_blocks_used"],
            "num_normal_segments": len(acc["normal_segment_means"]),
            "num_anomaly_events": len(acc["anomaly_segment_means"]),
            "num_boundaries": len(acc["boundary_means"]),
            "normal_pair_count": acc["normal_pair_count"],
            "anomaly_pair_count": acc["anomaly_pair_count"],
            "boundary_pair_count": acc["boundary_pair_count"],
            "fdr_direction": "between_over_within_higher_is_better",
            "variation_aggregation": "segment_event_boundary_macro",
        })
    return rows
