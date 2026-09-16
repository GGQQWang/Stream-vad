"""Offline event-macro Stage-AP for causal Stage-1 score representations."""

from __future__ import annotations

from typing import Sequence

import numpy as np


SPLIT_RULE = (
    "numpy.array_split over ordered event-window indices; remainder windows "
    "are assigned to earlier thirds"
)


def find_anomaly_events(frame_gt: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous positive frame intervals as half-open ``[start, end)``."""
    abnormal = np.asarray(frame_gt).reshape(-1) != 0
    padded = np.pad(abnormal.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def split_event_windows(window_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ordered event windows into deterministic early/middle/late thirds."""
    indices = np.asarray(window_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("window_indices must be one-dimensional")
    if len(indices) > 1 and np.any(indices[1:] <= indices[:-1]):
        raise ValueError("window_indices must be strictly increasing")
    early, middle, late = np.array_split(indices, 3)
    return early, middle, late


def deterministic_uniform_sample(indices: Sequence[int], count: int) -> np.ndarray:
    """Select ``count`` ordered positions spanning the full input deterministically."""
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if count < 0 or count > len(values):
        raise ValueError(f"count must be in [0, {len(values)}], got {count}")
    if count == 0:
        return values[:0]
    positions = np.floor(np.linspace(0, len(values) - 1, num=count)).astype(np.int64)
    return values[positions]


def _validate_window_timeline(frame_gt: np.ndarray, window_rows: Sequence[dict]) -> None:
    n_frames = len(frame_gt)
    if not window_rows:
        raise ValueError("Stage-AP requires at least one inference window")
    previous_end = 0
    previous_index = None
    for position, row in enumerate(window_rows):
        index = int(row["window_index"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if position == 0 and index != 0:
            raise ValueError(f"first inference window index is {index}, expected 0")
        if previous_index is not None and index != previous_index + 1:
            raise ValueError(
                f"inference window indices are not contiguous: got {index} after {previous_index}"
            )
        if start != previous_end:
            raise ValueError(
                f"inference window timeline is not contiguous: start={start}, expected {previous_end}"
            )
        if end <= start or end > n_frames:
            raise ValueError(f"invalid inference window [{start}, {end}) for n_frames={n_frames}")
        previous_index = index
        previous_end = end
    if previous_end != n_frames:
        raise ValueError(f"inference windows end at frame {previous_end}, expected {n_frames}")


def _adjacent_normal_windows(
    normal_windows: np.ndarray,
    boundary: int,
    direction: int,
    limit: int,
) -> np.ndarray:
    selected: list[int] = []
    index = boundary + direction
    while 0 <= index < len(normal_windows) and bool(normal_windows[index]) and len(selected) < limit:
        selected.append(index)
        index += direction
    if direction < 0:
        selected.reverse()
    return np.asarray(selected, dtype=np.int64)


def ranking_average_precision(
    query_hidden: np.ndarray,
    positive_hidden: np.ndarray,
    negative_hidden: np.ndarray,
) -> float:
    """Standard binary-relevance AP ranked by cosine similarity, without thresholds."""
    query = np.asarray(query_hidden, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive_hidden, dtype=np.float64)
    negative = np.asarray(negative_hidden, dtype=np.float64)
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive_hidden and negative_hidden must be rank-2")
    if len(positive) < 1 or len(negative) < 1:
        raise ValueError("AP requires at least one positive and one negative")
    candidates = np.concatenate([positive, negative], axis=0)
    if candidates.shape[1] != query.shape[0]:
        raise ValueError("query and candidate hidden dimensions do not match")
    query_norm = max(float(np.linalg.norm(query)), np.finfo(np.float64).eps)
    candidate_norms = np.maximum(
        np.linalg.norm(candidates, axis=1), np.finfo(np.float64).eps,
    )
    similarities = candidates @ query / (candidate_norms * query_norm)
    labels = np.concatenate([
        np.ones(len(positive), dtype=np.int64),
        np.zeros(len(negative), dtype=np.int64),
    ])
    if not np.isfinite(similarities).all():
        raise ValueError("cosine similarities must be finite")

    # Standard non-interpolated ranking AP, equivalent to
    # sklearn.metrics.average_precision_score for binary labels.  Candidates
    # with tied scores enter the ranked set together at one threshold.
    order = np.argsort(-similarities, kind="mergesort")
    sorted_scores = similarities[order]
    sorted_labels = labels[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(labels) - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_ends]
    retrieved = threshold_ends + 1
    precision = true_positives / retrieved
    recall = true_positives / len(positive)
    recall_gain = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_gain * precision))


def _transition_ap(
    query_indices: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    score_hidden: np.ndarray,
) -> float:
    positive = score_hidden[positive_indices]
    negative = score_hidden[negative_indices]
    values = [
        ranking_average_precision(score_hidden[index], positive, negative)
        for index in query_indices
    ]
    return float(np.mean(values))


def evaluate_video_stage_ap(
    *,
    video_id: str,
    frame_gt: np.ndarray,
    window_rows: Sequence[dict],
    score_hidden: np.ndarray,
) -> list[dict]:
    """Evaluate all frame-level anomaly events in one video's inference timeline.

    A window belongs to an event when its exact standard-AUC frame span overlaps
    that event.  A window overlapping two events is excluded from both events'
    stage features.  Pre/Post are the nearest consecutive all-normal windows and
    are scanned only as far as the maximum usable stage size.
    """
    gt = np.asarray(frame_gt).reshape(-1)
    hidden = np.asarray(score_hidden)
    _validate_window_timeline(gt, window_rows)
    if hidden.ndim != 2 or hidden.shape[0] != len(window_rows):
        raise ValueError(
            f"score_hidden must be [num_windows, hidden_dim], got {hidden.shape} "
            f"for {len(window_rows)} windows"
        )
    if hidden.shape[1] < 1 or not np.isfinite(hidden).all():
        raise ValueError("score_hidden must have a non-empty finite hidden dimension")

    events = find_anomaly_events(gt)
    frame_event = np.full(len(gt), -1, dtype=np.int64)
    for event_index, (start, end) in enumerate(events):
        frame_event[start:end] = event_index

    memberships: list[frozenset[int]] = []
    for row in window_rows:
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        ids = np.unique(frame_event[start:end])
        memberships.append(frozenset(int(value) for value in ids if value >= 0))
    normal_windows = np.asarray([not membership for membership in memberships], dtype=bool)

    records: list[dict] = []
    for event_index, (start, end) in enumerate(events):
        mapped = np.asarray(
            [index for index, membership in enumerate(memberships) if event_index in membership],
            dtype=np.int64,
        )
        exclusive = np.asarray(
            [index for index, membership in enumerate(memberships) if membership == {event_index}],
            dtype=np.int64,
        )
        base = {
            "video_id": video_id,
            "event_id": f"{video_id}:event_{event_index:04d}",
            "start": int(start),
            "end": int(end),
            "interval_convention": "frame_[start,end)",
            "window_range_convention": "inclusive_window_indices",
            "mapped_window_indices": [int(window_rows[i]["window_index"]) for i in mapped],
            "ambiguous_windows_excluded": int(len(mapped) - len(exclusive)),
            "split_rule": SPLIT_RULE,
            "m": 0,
            "ap_em": None,
            "ap_ml": None,
            "ap_el": None,
            "stage_ap": None,
            "pre_range": None,
            "post_range": None,
            "window_indices": None,
        }
        if len(exclusive) == 0:
            records.append({
                **base,
                "status": "skipped",
                "skip_reason": "no_unambiguous_event_windows",
            })
            continue

        early, middle, late = split_event_windows(exclusive)
        stage_limit = min(len(early), len(middle), len(late))
        if stage_limit < 1:
            records.append({
                **base,
                "status": "skipped",
                "skip_reason": "fewer_than_three_event_windows",
            })
            continue

        pre = _adjacent_normal_windows(normal_windows, int(mapped.min()), -1, stage_limit)
        post = _adjacent_normal_windows(normal_windows, int(mapped.max()), 1, stage_limit)
        m = min(len(early), len(middle), len(late), len(pre), len(post))
        if m < 1:
            missing = []
            if len(pre) == 0:
                missing.append("pre")
            if len(post) == 0:
                missing.append("post")
            records.append({
                **base,
                "status": "skipped",
                "skip_reason": "missing_adjacent_" + "_and_".join(missing) + "_normal",
            })
            continue

        sampled = {
            "pre": deterministic_uniform_sample(pre, m),
            "early": deterministic_uniform_sample(early, m),
            "middle": deterministic_uniform_sample(middle, m),
            "late": deterministic_uniform_sample(late, m),
            "post": deterministic_uniform_sample(post, m),
        }
        negative = np.concatenate([sampled["pre"], sampled["post"]])
        if len(negative) != 2 * m:
            raise AssertionError("Stage-AP candidate set must contain exactly 2m negatives")
        ap_em = _transition_ap(sampled["early"], sampled["middle"], negative, hidden)
        ap_ml = _transition_ap(sampled["middle"], sampled["late"], negative, hidden)
        ap_el = _transition_ap(sampled["early"], sampled["late"], negative, hidden)
        to_window_ids = lambda values: [int(window_rows[i]["window_index"]) for i in values]
        window_ids = {name: to_window_ids(values) for name, values in sampled.items()}
        records.append({
            **base,
            "status": "valid",
            "skip_reason": None,
            "m": int(m),
            "ap_em": ap_em,
            "ap_ml": ap_ml,
            "ap_el": ap_el,
            "stage_ap": float((ap_em + ap_ml + ap_el) / 3.0),
            "pre_range": [window_ids["pre"][0], window_ids["pre"][-1]],
            "post_range": [window_ids["post"][0], window_ids["post"][-1]],
            "window_indices": window_ids,
            "num_positive_candidates_per_query": int(m),
            "num_negative_candidates_per_query": int(2 * m),
        })
    return records


def aggregate_event_metrics(records: Sequence[dict]) -> dict:
    """Compute dataset metrics as event-level macro averages."""
    valid = [record for record in records if record.get("status") == "valid"]
    skipped = [record for record in records if record.get("status") != "valid"]
    if not valid:
        em_ap = ml_ap = el_ap = stage_ap = None
    else:
        em_ap = float(np.mean([record["ap_em"] for record in valid]))
        ml_ap = float(np.mean([record["ap_ml"] for record in valid]))
        el_ap = float(np.mean([record["ap_el"] for record in valid]))
        stage_ap = float(np.mean([record["stage_ap"] for record in valid]))
    return {
        "num_valid_events": len(valid),
        "num_skipped_events": len(skipped),
        "em_ap": em_ap,
        "ml_ap": ml_ap,
        "el_ap": el_ap,
        "stage_ap": stage_ap,
        "feature": "score_hidden",
        "split": "numpy_array_split_thirds",
        "split_rule": SPLIT_RULE,
        "negative_source": "adjacent_pre_post_normal",
        "aggregation": "event_macro",
        "ap": "standard_non_interpolated_ranking_ap_ties_grouped",
        "score_range": "0_to_1",
    }
