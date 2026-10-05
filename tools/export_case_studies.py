"""Export GT-selected qualitative cases from completed Stage-1 inference."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import numpy as np


BINS = ("short", "medium", "long")
INDEX_FIELDS = (
    "case_id", "video_id", "event_index", "duration_bin", "event_duration_sec",
    "event_start_sec", "event_end_sec", "selection_mode", "seed", "output_path",
    "case_role", "video_standard_auc", "auc_rank", "selection_basis",
)
CANDIDATE_VIDEO_FIELDS = (
    "video_id", "standard_auc", "standard_ap", "causal_auc", "causal_ap",
    "auc_rank", "num_gt_events", "pool_role",
)
AUC_SELECTION_BASIS = "per-video standard_auc from existing metrics.json"
TIMELINE_FIELDS = (
    "video_id", "case_id", "window_index", "start_frame", "end_frame",
    "start_sec", "end_sec", "score_prob", "gt_overlap", "gt_at_window_end",
)
WINDOW_FIELDS = (
    "video_id", "window_index", "start_frame", "end_frame", "start_sec",
    "end_sec", "score_prob",
)


@dataclass(frozen=True)
class Event:
    video_id: str
    event_index: int
    start_frame: int
    end_frame: int
    fps: float
    n_frames: int

    @property
    def case_id(self) -> str:
        return f"{self.video_id}__event{self.event_index:02d}"

    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps


@dataclass(frozen=True)
class SelectedCase:
    event: Event
    duration_bin: str
    selection_mode: str
    reason: str
    random_rank: int | None = None
    case_role: str = ""
    video_standard_auc: float | None = None
    auc_rank: int | None = None
    success_pool_size: int | None = None
    failure_pool_bottom_k: int | None = None
    sampling_seed: int | None = None


def normalize_video_id(value: str | Path) -> str:
    video_id = str(value)
    return video_id[:-4] if video_id.lower().endswith(".mp4") else video_id


def _logical_id(entry: dict) -> str:
    for key in ("video_id", "id", "name"):
        if key in entry:
            return normalize_video_id(entry[key])
    for key in ("video", "video_path", "path", "filename"):
        if key in entry:
            return normalize_video_id(Path(str(entry[key])).name)
    raise ValueError(f"manifest entry has no video id: {entry!r}")


def load_manifest(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        entries = raw.items()
    elif isinstance(raw, list):
        entries = ((_logical_id(entry), entry) for entry in raw)
    else:
        raise ValueError("manifest must be a video-id mapping or list of entries")
    manifest = {}
    for raw_id, entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid manifest entry for {raw_id!r}")
        video_id = normalize_video_id(raw_id)
        if not video_id or video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
            raise ValueError(f"unsafe logical video_id: {video_id!r}")
        if video_id in manifest:
            raise ValueError(f"duplicate normalized video_id: {video_id}")
        n_frames = int(entry.get("n_frames", entry.get("num_frames", 0)))
        fps = float(entry.get("fps", 0))
        if n_frames <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid n_frames/fps for {video_id}: {n_frames}, {fps}")
        manifest[video_id] = {**entry, "n_frames": n_frames, "fps": fps}
    if not manifest:
        raise ValueError("empty test manifest")
    return manifest


def load_gt(gt_root: str | Path, video_id: str, n_frames: int) -> np.ndarray:
    path = Path(gt_root) / f"{video_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing frame-level GT: {path}")
    gt = np.atleast_1d(np.loadtxt(path, dtype=np.int64))
    if gt.ndim != 1 or gt.size != n_frames or not np.isin(gt, (0, 1)).all():
        raise ValueError(f"{video_id}: GT must contain {n_frames} binary frame labels")
    return gt


def contiguous_events(gt: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(gt)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("GT must be a one-dimensional binary frame-label array")
    padded = np.pad(labels.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def build_candidates(manifest: dict[str, dict], gt_root: str | Path) -> list[Event]:
    events = []
    for video_id, meta in sorted(manifest.items()):
        gt = load_gt(gt_root, video_id, meta["n_frames"])
        for event_index, (start, end) in enumerate(contiguous_events(gt)):
            events.append(Event(video_id, event_index, start, end, meta["fps"], meta["n_frames"]))
    return events


def duration_bins(events: list[Event]) -> tuple[dict[str, str], tuple[float, float]]:
    if not events:
        raise ValueError("no abnormal GT events in manifest")
    q1, q2 = np.quantile([event.duration_sec for event in events], [1 / 3, 2 / 3])
    bins = {}
    for event in events:
        duration = event.duration_sec
        bins[event.case_id] = "short" if duration <= q1 else "medium" if duration <= q2 else "long"
    return bins, (float(q1), float(q2))


def select_cases(
    events: list[Event], bins: dict[str, str], *, mode: str, num_cases: int,
    seed: int, selectors: list[str] | None = None,
) -> tuple[list[SelectedCase], dict]:
    selectors = selectors or []
    by_case = {event.case_id: event for event in events}
    by_video: dict[str, list[Event]] = {}
    for event in events:
        by_video.setdefault(event.video_id, []).append(event)
    if mode == "manual":
        if not selectors:
            raise ValueError("manual selection requires --video-ids or --video-id-file")
        selected = []
        seen = set()
        for selector in selectors:
            key = normalize_video_id(selector)
            matches = by_video.get(key) if key in by_video else [by_case[key]] if key in by_case else None
            if matches is None:
                raise ValueError(f"no GT abnormal event matches manual selector {selector!r}")
            for event in matches:
                if event.case_id not in seen:
                    selected.append(SelectedCase(event, bins[event.case_id], mode, "user-supplied selector"))
                    seen.add(event.case_id)
        return selected, {"requested": len(selected), "allocated": None}
    if selectors:
        raise ValueError("--video-ids/--video-id-file require --selection-mode manual")
    if mode == "all":
        selected = [
            SelectedCase(event, bins[event.case_id], mode, "all GT abnormal events")
            for event in sorted(events, key=lambda item: item.case_id)
        ]
        return selected, {"requested": len(selected), "allocated": None}
    if mode != "stratified-random" or num_cases <= 0:
        raise ValueError("stratified-random requires --num-cases > 0")

    groups = {
        name: sorted((event for event in events if bins[event.case_id] == name), key=lambda item: item.case_id)
        for name in BINS
    }
    desired = {name: num_cases // 3 + (i < num_cases % 3) for i, name in enumerate(BINS)}
    allocated = {name: min(desired[name], len(groups[name])) for name in BINS}
    remaining = num_cases - sum(allocated.values())
    while remaining:
        advanced = False
        for name in BINS:
            if allocated[name] < len(groups[name]):
                allocated[name] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
        if not advanced:
            break

    rng = np.random.default_rng(seed)
    selected = []
    for name in BINS:
        for rank, index in enumerate(rng.permutation(len(groups[name])), start=1):
            if rank > allocated[name]:
                break
            selected.append(SelectedCase(
                groups[name][int(index)], name, mode,
                "GT-duration tertile; seeded random permutation", rank,
            ))
    audit = {
        "requested": num_cases,
        "candidate_counts": {name: len(groups[name]) for name in BINS},
        "initial_quota": desired,
        "allocated": allocated,
        "initial_shortage": {name: max(0, desired[name] - len(groups[name])) for name in BINS},
        "unfilled_total": remaining,
    }
    return selected, audit


def load_video_metrics(path: str | Path, events: list[Event], manifest: dict[str, dict]) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict) or not isinstance(metrics.get("videos"), list):
        raise ValueError(f"{path}: expected metrics['videos'] list")
    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event.video_id] = event_counts.get(event.video_id, 0) + 1
    records = []
    seen = set()
    for video in metrics["videos"]:
        if not isinstance(video, dict) or "video_id" not in video:
            raise ValueError(f"{path}: invalid per-video metrics entry")
        video_id = normalize_video_id(video["video_id"])
        if video_id in seen:
            raise ValueError(f"{path}: duplicate video_id in metrics: {video_id}")
        seen.add(video_id)
        if video_id not in manifest or not event_counts.get(video_id):
            continue
        auc = video.get("standard_auc")
        if auc is None:
            continue
        if isinstance(auc, bool) or not isinstance(auc, (int, float)) or not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError(f"{path}: invalid standard_auc for {video_id}: {auc!r}")
        records.append({
            "video_id": video_id, "standard_auc": float(auc),
            "standard_ap": video.get("standard_ap"),
            "causal_auc": video.get("causal_auc"), "causal_ap": video.get("causal_ap"),
            "num_gt_events": event_counts[video_id],
        })
    records.sort(key=lambda item: (-item["standard_auc"], item["video_id"]))
    for rank, record in enumerate(records, start=1):
        record["auc_rank"] = rank
    return records


def select_high_auc_cases(
    events: list[Event], records: list[dict], *, infer_dir: Path, num_cases: int,
    seed: int, auc_top_k: int | None, auc_min: float | None,
    num_failures: int, failure_bottom_k: int, failure_seed: int,
) -> tuple[list[SelectedCase], dict, list[dict]]:
    if num_cases < 0 or num_failures < 0 or (auc_top_k is not None and auc_top_k <= 0):
        raise ValueError("num-cases/num-failures must be nonnegative; auc-top-k must be positive")
    if num_failures and failure_bottom_k <= 0:
        raise ValueError("failure-bottom-k must be positive when selecting failures")
    if auc_min is not None and (not math.isfinite(auc_min) or not 0 <= auc_min <= 1):
        raise ValueError("auc-min must be within [0, 1]")

    success_pool = [record for record in records if auc_min is None or record["standard_auc"] >= auc_min]
    if auc_top_k is not None:
        success_pool = success_pool[:auc_top_k]
    # Define the bottom-K on all eligible abnormal videos, before disjointness handling.
    failure_pool = (
        sorted(records, key=lambda item: (item["standard_auc"], item["video_id"]))[:failure_bottom_k]
        if num_failures else []
    )
    success_ids = {record["video_id"] for record in success_pool}
    missing = sorted({record["video_id"] for record in success_pool + failure_pool
                      if not (infer_dir / f"{record['video_id']}_window_scores.csv").is_file()})
    if missing:
        raise FileNotFoundError(
            f"selected AUC candidate pool lacks window-score CSV; selection aborted: {missing}"
        )

    events_by_video: dict[str, list[Event]] = {}
    for event in events:
        events_by_video.setdefault(event.video_id, []).append(event)
    success_events = [event for record in success_pool for event in events_by_video[record["video_id"]]]
    if not success_events and not num_failures:
        raise ValueError("no calculable abnormal videos satisfy the high-AUC selection filters")
    selected: list[SelectedCase] = []
    success_quantiles = None
    if success_events:
        bins, success_quantiles = duration_bins(success_events)
        if num_cases:
            sampled, _ = select_cases(success_events, bins, mode="stratified-random", num_cases=num_cases, seed=seed)
        else:
            sampled, _ = select_cases(success_events, bins, mode="all", num_cases=0, seed=seed)
        by_id = {record["video_id"]: record for record in success_pool}
        selected = [replace(
            case, selection_mode="high-auc-pool", case_role="success",
            reason="high per-video standard_auc; duration-stratified seeded sampling" if num_cases else
                   "high per-video standard_auc; all candidate events",
            video_standard_auc=by_id[case.event.video_id]["standard_auc"],
            auc_rank=by_id[case.event.video_id]["auc_rank"], success_pool_size=len(success_pool),
            sampling_seed=seed,
        ) for case in sampled]

    # Never export the same logical case as both success and failure.
    failure_options = [record for record in failure_pool if record["video_id"] not in success_ids]
    if num_failures > len(failure_options):
        raise ValueError(
            f"bottom-K failure pool has only {len(failure_options)} videos outside the success pool; "
            f"cannot select {num_failures} distinct failures"
        )
    rng = np.random.default_rng(failure_seed)
    indices = rng.permutation(len(failure_options))[:num_failures]
    failure_bins = {}
    if events:
        failure_bins, _ = duration_bins(events)
    for index in indices:
        record = failure_options[int(index)]
        video_events = events_by_video[record["video_id"]]
        median = float(np.median([event.duration_sec for event in video_events]))
        event = min(video_events, key=lambda item: (abs(item.duration_sec - median), item.event_index))
        selected.append(SelectedCase(
            event, failure_bins[event.case_id], "high-auc-pool",
            "seeded random video from bottom-K per-video standard_auc; median-duration GT event",
            case_role="failure", video_standard_auc=record["standard_auc"],
            auc_rank=record["auc_rank"], failure_pool_bottom_k=failure_bottom_k,
            sampling_seed=failure_seed,
        ))

    selected.sort(key=lambda case: (
        case.case_role != "success",
        -case.video_standard_auc if case.case_role == "success" else case.video_standard_auc,
        case.event.case_id,
    ))
    candidate_rows = []
    failure_ids = {record["video_id"] for record in failure_pool}
    for record in records:
        video_id = record["video_id"]
        candidate_rows.append({
            **record,
            "pool_role": "success_candidate" if video_id in success_ids else
                         "failure_candidate" if video_id in failure_ids else "unused",
        })
    audit = {
        "selection_mode": "high-auc-pool", "selection_basis": AUC_SELECTION_BASIS,
        "primary_filter": "high per-video standard_auc",
        "secondary_sampling": "duration-stratified seeded sampling" if num_cases else "all candidate events",
        "auc_top_k": auc_top_k, "auc_min": auc_min,
        "success_video_pool_size": len(success_pool), "success_event_pool_size": len(success_events),
        "success_duration_quantiles_sec": success_quantiles,
        "num_selected_success": sum(case.case_role == "success" for case in selected),
        "num_failures": sum(case.case_role == "failure" for case in selected),
        "failure_bottom_k": failure_bottom_k, "seed": seed, "failure_seed": failure_seed,
        "success_candidate_videos": success_pool, "failure_pool_videos": failure_pool,
        "failure_pool_excluded_success_video_ids": sorted(success_ids & failure_ids),
    }
    return selected, audit, candidate_rows


def temporal_anchors(
    event: Event, *, pre_sec: float, post_sec: float, frames_per_case: int,
) -> tuple[list[tuple[str, int]], list[dict]]:
    if pre_sec < 0 or post_sec < 0 or frames_per_case < 3:
        raise ValueError("pre/post seconds must be nonnegative and frames_per_case >= 3")
    start, end = event.start_frame, event.end_frame
    anchors: list[tuple[str, int]] = []
    if pre_sec > 0 and start > 0:
        anchors.append(("pre", max(0, start - max(1, round(pre_sec * event.fps / 2)))))
    progress = np.linspace(0, end - start - 1, frames_per_case - 2)
    default_names = ("start", "early", "middle", "late", "end")
    for i, offset in enumerate(progress):
        if frames_per_case == 7:
            label = default_names[i]
        else:
            label = "start" if i == 0 else "end" if i == len(progress) - 1 else f"event_{i:02d}"
        anchors.append((label, start + int(round(float(offset)))))
    if post_sec > 0 and end < event.n_frames:
        anchors.append(("post", min(event.n_frames - 1, end - 1 + max(1, round(post_sec * event.fps / 2)))))
    kept, duplicates, seen = [], [], set()
    for label, frame in anchors:
        if frame in seen:
            duplicates.append({"anchor": label, "frame_index": frame})
        else:
            kept.append((label, frame))
            seen.add(frame)
    return kept, duplicates


def context_interval(event: Event, *, pre_sec: float, post_sec: float) -> tuple[float, float]:
    if pre_sec < 0 or post_sec < 0:
        raise ValueError("pre/post seconds must be nonnegative")
    return max(0.0, event.start_sec - pre_sec), min(event.n_frames / event.fps, event.end_sec + post_sec)


def plot_context_interval(
    event: Event, frame_context: tuple[float, float], plot_range: str,
) -> tuple[float, float]:
    if plot_range == "full":
        return 0.0, event.n_frames / event.fps
    if plot_range == "local":
        return frame_context
    raise ValueError(f"unknown plot range: {plot_range}")


def resolve_video_path(video_root: str | Path, video_id: str, entry: dict) -> Path:
    root = Path(video_root)
    preferred = root / f"{video_id}.mp4"
    if preferred.is_file():
        return preferred
    for key in ("video_path", "video", "path", "filename"):
        if key in entry:
            candidate = Path(str(entry[key]))
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"video for {video_id} not found at {preferred} or manifest path")


def read_window_scores(path: str | Path, *, video_id: str, n_frames: int, fps: float) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(WINDOW_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"missing required window-score columns in {path}")
        rows = []
        for raw in reader:
            row = {
                "video_id": raw["video_id"], "window_index": int(raw["window_index"]),
                "start_frame": int(raw["start_frame"]), "end_frame": int(raw["end_frame"]),
                "start_sec": float(raw["start_sec"]), "end_sec": float(raw["end_sec"]),
                "score_prob": float(raw["score_prob"]),
            }
            start, end = row["start_frame"], row["end_frame"]
            if row["video_id"] != video_id or not 0 <= start < end <= n_frames:
                raise ValueError(f"invalid window identity/range: {row}")
            if not math.isfinite(row["score_prob"]) or not 0 <= row["score_prob"] <= 1:
                raise ValueError(f"invalid score_prob: {row}")
            if abs(row["start_sec"] - start / fps) > 1e-5 or abs(row["end_sec"] - end / fps) > 1e-5:
                raise ValueError(f"window timestamps disagree with manifest fps: {row}")
            rows.append(row)
    rows.sort(key=lambda item: item["window_index"])
    if not rows or rows[0]["start_frame"] != 0 or rows[-1]["end_frame"] != n_frames:
        raise ValueError(f"incomplete window-score timeline for {video_id}")
    for previous, current in zip(rows, rows[1:]):
        if previous["end_frame"] != current["start_frame"] or previous["window_index"] + 1 != current["window_index"]:
            raise ValueError(f"non-contiguous window-score timeline for {video_id}")
    return rows


def case_timeline(
    case: SelectedCase, rows: list[dict], gt: np.ndarray,
) -> list[dict]:
    out = []
    for row in rows:
        start, end = row["start_frame"], row["end_frame"]
        out.append({
            **row,
            "case_id": case.event.case_id,
            "gt_overlap": float(gt[start:end].mean()),
            "gt_at_window_end": int(gt[end - 1]),
        })
    return out


def score_points(
    infer_dir: str | Path, case: SelectedCase, rows: list[dict],
    plot_context: tuple[float, float], mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    event = case.event
    if mode == "window-end":
        points = [row for row in rows if plot_context[0] <= row["end_sec"] <= plot_context[1]]
        return (
            np.array([row["end_sec"] for row in points], dtype=float),
            np.array([row["score_prob"] for row in points], dtype=float),
            "window score emitted at window end",
        )
    if mode not in {"causal-frame", "standard-frame"}:
        raise ValueError(f"unknown score mode: {mode}")
    suffix = "causal_frame_scores" if mode == "causal-frame" else "standard_frame_scores"
    values = np.load(Path(infer_dir) / f"{event.video_id}_{suffix}.npy", allow_pickle=False)
    if values.shape != (event.n_frames,):
        raise ValueError(f"{event.video_id}: {suffix} shape {values.shape} does not match GT")
    mask = np.ones(event.n_frames, dtype=bool)
    if mode == "causal-frame":
        mask = np.load(Path(infer_dir) / f"{event.video_id}_causal_valid_mask.npy", allow_pickle=False)
        if mask.shape != (event.n_frames,) or mask.dtype != np.bool_:
            raise ValueError(f"{event.video_id}: invalid causal_valid_mask")
    times = np.arange(event.n_frames) / event.fps
    mask &= (times >= plot_context[0]) & (times <= plot_context[1])
    if not np.isfinite(values[mask]).all():
        raise ValueError(f"{event.video_id}: non-finite {mode} score in valid region")
    note = (
        "causal frame mapping; score available only after source window end"
        if mode == "causal-frame" else
        "standard frame expansion; not emission-time causal alignment"
    )
    return times[mask], values[mask], note


def decode_selected_frames(video_path: Path, indices: list[int], n_frames: int) -> list[np.ndarray]:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError("decord is required only when exporting video frames") from exc
    reader = VideoReader(str(video_path), ctx=cpu(0))
    if len(reader) != n_frames:
        raise ValueError(f"{video_path}: decoded {len(reader)} frames, manifest expects {n_frames}")
    return list(reader.get_batch(indices).asnumpy())


def _draw_curve(ax, *, case: SelectedCase, x: np.ndarray, y: np.ndarray,
                all_events: list[tuple[int, int]], plot_context: tuple[float, float],
                score_mode: str):
    event = case.event
    other_label_used = False
    selected_found = False
    for start, end in all_events:
        selected = (start, end) == (event.start_frame, event.end_frame)
        selected_found |= selected
        left, right = max(plot_context[0], start / event.fps), min(plot_context[1], end / event.fps)
        if right > left:
            label = "Selected GT event" if selected else "Other GT anomaly interval" if not other_label_used else None
            ax.axvspan(left, right, color="#c6635a", alpha=0.20 if selected else 0.10, label=label)
            if not selected:
                other_label_used = True
    if not selected_found:
        raise ValueError(f"selected event {event.case_id} is absent from video GT intervals")
    ax.axvline(event.start_sec, color="#b44a42", linewidth=0.9, linestyle="--")
    ax.axvline(event.end_sec, color="#b44a42", linewidth=0.9, linestyle="--")
    if len(x):
        if score_mode == "window-end":
            ax.step(x, y, where="post", color="#176b8a", linewidth=1.5,
                    marker="o", markersize=2.5, label="Anomaly score")
            if x[-1] < plot_context[1]:
                ax.hlines(y[-1], x[-1], plot_context[1], color="#176b8a", linewidth=1.5)
        else:
            ax.plot(x, y, color="#176b8a", linewidth=1.2, label="Anomaly score")
    ax.set_xlim(*plot_context)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Anomaly score")
    ax.grid(axis="y", color="#e3e5e8", linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def render_case(
    case_dir: Path, case: SelectedCase, anchors: list[tuple[str, int]],
    frames: list[np.ndarray], x: np.ndarray, y: np.ndarray,
    all_events: list[tuple[int, int]], plot_context: tuple[float, float], dpi: int,
    score_mode: str,
    local_zoom: tuple[np.ndarray, np.ndarray, tuple[float, float]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": "white"})
    frame_dir = case_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, ((label, _), frame) in enumerate(zip(anchors, frames)):
        plt.imsave(frame_dir / f"frame_{i:02d}_{label}.png", frame)

    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    _draw_curve(ax, case=case, x=x, y=y, all_events=all_events,
                plot_context=plot_context, score_mode=score_mode)
    fig.savefig(case_dir / "score_curve.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(case_dir / "score_curve.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if local_zoom is not None:
        local_x, local_y, local_context = local_zoom
        fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
        _draw_curve(ax, case=case, x=local_x, y=local_y, all_events=all_events,
                    plot_context=local_context, score_mode=score_mode)
        fig.savefig(case_dir / "score_curve_local.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(case_dir / "score_curve_local.png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig = plt.figure(figsize=(max(7.2, len(frames) * 1.45), 4.7), layout="constrained")
    grid = fig.add_gridspec(2, len(frames), height_ratios=(1.15, 1.0))
    for i, ((_, frame_index), frame) in enumerate(zip(anchors, frames)):
        ax_frame = fig.add_subplot(grid[0, i])
        ax_frame.imshow(frame)
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        for spine in ax_frame.spines.values():
            spine.set_visible(False)
        ax_frame.set_xlabel(f"{frame_index / case.event.fps:.2f} s", fontsize=8)
    ax_curve = fig.add_subplot(grid[1, :])
    _draw_curve(ax_curve, case=case, x=x, y=y, all_events=all_events,
                plot_context=plot_context, score_mode=score_mode)
    fig.savefig(case_dir / "case_panel.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(case_dir / "case_panel.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def git_worktree_dirty() -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root,
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def case_metadata(
    case: SelectedCase, *, seed: int, score_mode: str, plot_range: str,
    frame_context: tuple[float, float], plot_context: tuple[float, float],
    num_gt_events_in_video: int, local_zoom_exported: bool,
    anchors: list[tuple[str, int]], duplicates: list[dict], source_window_score_csv: Path,
    source_gt: Path, source_video: Path, commit: str, score_note: str,
    worktree_dirty: bool = False,
) -> dict:
    event = case.event
    metadata = {
        "case_id": event.case_id,
        "video_id": event.video_id,
        "event_index": event.event_index,
        "fps": event.fps,
        "n_frames": event.n_frames,
        "event_start_frame": event.start_frame,
        "event_end_frame": event.end_frame,
        "event_start_sec": event.start_sec,
        "event_end_sec": event.end_sec,
        "event_duration_sec": event.duration_sec,
        "duration_bin": case.duration_bin,
        "selection_mode": case.selection_mode,
        "selection_reason": case.reason,
        "seed": case.sampling_seed if case.sampling_seed is not None else seed,
        "score_mode": score_mode,
        "score_mode_note": score_note,
        "gt_overlap_definition": "fraction of GT-positive frames in [start_frame, end_frame)",
        "gt_at_window_end_definition": "GT label of the last observed frame (end_frame - 1)",
        "plot_range": plot_range,
        "frame_context_start_sec": frame_context[0],
        "frame_context_end_sec": frame_context[1],
        "plot_context_start_sec": plot_context[0],
        "plot_context_end_sec": plot_context[1],
        "num_gt_events_in_video": num_gt_events_in_video,
        "selected_event_index": event.event_index,
        "local_zoom_exported": local_zoom_exported,
        "selected_frame_indices": [index for _, index in anchors],
        "selected_frame_times": [index / event.fps for _, index in anchors],
        "selected_frame_anchor_labels": [label for label, _ in anchors],
        "dropped_duplicate_anchors": duplicates,
        "source_window_score_csv": str(source_window_score_csv),
        "source_gt": str(source_gt),
        "source_video": str(source_video),
        "git_commit": commit,
        "git_worktree_dirty": worktree_dirty,
    }
    if case.random_rank is not None:
        metadata["selection_rank_within_random_permutation"] = case.random_rank
    if case.selection_mode == "high-auc-pool":
        metadata.update({
            "case_role": case.case_role, "selection_basis": AUC_SELECTION_BASIS,
            "video_standard_auc": case.video_standard_auc, "auc_rank": case.auc_rank,
        })
        if case.case_role == "success":
            metadata.update({
                "success_pool_rank": case.auc_rank, "success_pool_size": case.success_pool_size,
                "success_selection_reason": case.reason,
            })
        else:
            metadata.update({
                "failure_pool_bottom_k": case.failure_pool_bottom_k,
                "failure_selection_reason": case.reason,
            })
    return metadata


def index_row(case: SelectedCase, *, seed: int, output_path: Path) -> dict:
    event = case.event
    return {
        "case_id": event.case_id, "video_id": event.video_id,
        "event_index": event.event_index, "duration_bin": case.duration_bin,
        "event_duration_sec": event.duration_sec, "event_start_sec": event.start_sec,
        "event_end_sec": event.end_sec, "selection_mode": case.selection_mode,
        "seed": case.sampling_seed if case.sampling_seed is not None else seed,
        "case_role": case.case_role, "video_standard_auc": case.video_standard_auc,
        "auc_rank": case.auc_rank,
        "selection_basis": AUC_SELECTION_BASIS if case.selection_mode == "high-auc-pool" else "",
    }


def write_gallery(path: Path, index_rows: list[dict]) -> None:
    cards = []
    for row in sorted(index_rows, key=lambda item: (
        item["case_role"] == "failure",
        -item["video_standard_auc"] if item["case_role"] == "success" else
        item["video_standard_auc"] if item["case_role"] == "failure" else 0,
    )):
        case_id = html.escape(str(row["case_id"]))
        video_id = html.escape(str(row["video_id"]))
        href = quote(str(row["case_id"]), safe="")
        duration = float(row["event_duration_sec"])
        start, end = float(row["event_start_sec"]), float(row["event_end_sec"])
        role = html.escape(str(row["case_role"]))
        role_prefix = role + " · " if role else ""
        auc = row["video_standard_auc"]
        auc_label = f" · standard AUC {float(auc):.3f}" if auc is not None else ""
        cards.append(
            '<article><a href="' + href + '/case_panel.png"><img src="' + href +
            '/case_panel.png" alt="' + case_id + ' case panel"></a>' +
            '<div class="details"><strong>' + role_prefix + case_id + '</strong><span>' + video_id +
            auc_label + f' · {duration:.2f} s · {html.escape(str(row["duration_bin"]))}' +
            f' · {start:.2f}–{end:.2f} s</span><a href="' + href +
            '/case_panel.pdf">PDF</a></div></article>'
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Case studies</title>'
        '<style>body{font:14px system-ui,sans-serif;margin:24px;background:#fff;color:#20242a}'
        'main{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:20px}'
        'article{border:1px solid #d9dfe4;border-radius:6px;overflow:hidden}'
        'img{display:block;width:100%;height:auto}.details{padding:10px 12px;display:grid;gap:5px}'
        'span{color:#59616b}a{color:#176b8a;text-decoration:none}a:hover{text-decoration:underline}'
        '</style><main>' + "".join(cards) + '</main></html>'
    )
    path.write_text(document, encoding="utf-8")


def export_case(
    case: SelectedCase, *, infer_dir: Path, video_root: Path, gt_root: Path,
    manifest: dict[str, dict], output_dir: Path, seed: int, score_mode: str,
    pre_sec: float, post_sec: float, frames_per_case: int, dpi: int, commit: str,
    worktree_dirty: bool, plot_range: str, export_local_zoom: bool,
) -> dict:
    event = case.event
    case_dir = output_dir / event.case_id
    gt_path = gt_root / f"{event.video_id}.txt"
    score_csv = infer_dir / f"{event.video_id}_window_scores.csv"
    video_path = resolve_video_path(video_root, event.video_id, manifest[event.video_id])
    gt = load_gt(gt_root, event.video_id, event.n_frames)
    rows = read_window_scores(score_csv, video_id=event.video_id, n_frames=event.n_frames, fps=event.fps)
    frame_context = context_interval(event, pre_sec=pre_sec, post_sec=post_sec)
    plot_context = plot_context_interval(event, frame_context, plot_range)
    all_events = contiguous_events(gt)
    anchors, duplicates = temporal_anchors(
        event, pre_sec=pre_sec, post_sec=post_sec, frames_per_case=frames_per_case,
    )
    x, y, score_note = score_points(infer_dir, case, rows, plot_context, score_mode)
    local_zoom = None
    if export_local_zoom:
        local_x, local_y, _ = score_points(infer_dir, case, rows, frame_context, score_mode)
        local_zoom = (local_x, local_y, frame_context)
    frames = decode_selected_frames(video_path, [index for _, index in anchors], event.n_frames)
    case_dir.mkdir(parents=True, exist_ok=True)
    render_case(case_dir, case, anchors, frames, x, y, all_events,
                plot_context, dpi, score_mode, local_zoom=local_zoom)
    metadata = case_metadata(
        case, seed=seed, score_mode=score_mode, plot_range=plot_range,
        frame_context=frame_context, plot_context=plot_context,
        num_gt_events_in_video=len(all_events), local_zoom_exported=export_local_zoom,
        anchors=anchors, duplicates=duplicates, source_window_score_csv=score_csv,
        source_gt=gt_path, source_video=video_path, commit=commit, score_note=score_note,
        worktree_dirty=worktree_dirty,
    )
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_csv(case_dir / "timeline.csv", TIMELINE_FIELDS, case_timeline(case, rows, gt))
    return index_row(case, seed=seed, output_path=case_dir)


def _selectors_from_args(args) -> list[str]:
    selectors = list(args.video_ids or [])
    if args.video_id_file:
        for line in Path(args.video_id_file).read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                selectors.append(value)
    return selectors


def main(argv: list[str] | None = None) -> list[SelectedCase]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", default=[])
    parser.add_argument("--video-id-file", type=Path)
    parser.add_argument("--selection-mode", choices=("manual", "stratified-random", "high-auc-pool", "all"), default="stratified-random")
    parser.add_argument("--num-cases", type=int, default=12,
                        help="high-auc-pool: 0 exports every success event in the candidate pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auc-top-k", type=int, help="top N videos by existing per-video standard AUC")
    parser.add_argument("--auc-min", type=float, help="minimum existing per-video standard AUC")
    parser.add_argument("--num-failures", type=int, default=0, help="additional failure videos to sample")
    parser.add_argument("--failure-bottom-k", type=int, default=30,
                        help="failure sampling pool size among lowest-AUC abnormal videos")
    parser.add_argument("--failure-seed", type=int, help="defaults to --seed")
    parser.add_argument("--pre-sec", type=float, default=3.0)
    parser.add_argument("--post-sec", type=float, default=3.0)
    parser.add_argument("--plot-range", choices=("full", "local"), default="full",
                        help="score curve range; frame anchors always remain event-centered")
    parser.add_argument("--export-local-zoom", action="store_true",
                        help="also save a score-only local zoom around the selected event")
    parser.add_argument("--score-mode", choices=("window-end", "causal-frame", "standard-frame"), default="window-end")
    parser.add_argument("--frames-per-case", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.pre_sec < 0 or args.post_sec < 0 or args.frames_per_case < 3 or args.dpi <= 0:
        parser.error("pre/post seconds must be nonnegative; frames-per-case >= 3; dpi > 0")
    if args.selection_mode != "high-auc-pool" and any((
        args.auc_top_k is not None, args.auc_min is not None, args.num_failures,
        args.failure_bottom_k != 30, args.failure_seed is not None,
    )):
        parser.error("AUC pool options require --selection-mode high-auc-pool")
    manifest = load_manifest(args.manifest)
    candidates = build_candidates(manifest, args.gt_root)
    bins, quantiles = duration_bins(candidates)
    selectors = _selectors_from_args(args)
    if args.selection_mode == "high-auc-pool" and selectors:
        parser.error("--video-ids/--video-id-file require --selection-mode manual")
    if args.selection_mode in {"stratified-random", "all"} and not args.dry_run:
        # Missing inference output is an error, never an implicit case-selection filter.
        missing = sorted({event.video_id for event in candidates
                          if not (args.infer_dir / f"{event.video_id}_window_scores.csv").is_file()})
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} GT-positive videos lack window-score CSV; "
                f"selection aborted without filtering candidates: {missing[:20]}"
            )
    candidate_rows = None
    if args.selection_mode == "high-auc-pool":
        metrics_path = args.infer_dir / "metrics.json"
        records = load_video_metrics(metrics_path, candidates, manifest)
        selected, audit, candidate_rows = select_high_auc_cases(
            candidates, records, infer_dir=args.infer_dir, num_cases=args.num_cases,
            seed=args.seed, auc_top_k=args.auc_top_k, auc_min=args.auc_min,
            num_failures=args.num_failures, failure_bottom_k=args.failure_bottom_k,
            failure_seed=args.seed if args.failure_seed is None else args.failure_seed,
        )
    else:
        selected, audit = select_cases(
            candidates, bins, mode=args.selection_mode, num_cases=args.num_cases,
            seed=args.seed, selectors=selectors,
        )
    selection = {
        "selection_mode": args.selection_mode,
        "selection_basis": AUC_SELECTION_BASIS if args.selection_mode == "high-auc-pool" else
                           "GT event duration tertiles only; no model predictions or performance metrics",
        "seed": args.seed,
        "rng": "numpy.random.default_rng",
        "manifest": str(args.manifest),
        "gt_root": str(args.gt_root),
        "infer_dir": str(args.infer_dir),
        "duration_quantiles_sec": {"one_third": quantiles[0], "two_thirds": quantiles[1]},
        "num_gt_events": len(candidates),
        "num_selected": len(selected),
        "audit": audit,
        "selected": [
            {
                "case_id": case.event.case_id, "video_id": case.event.video_id,
                "event_index": case.event.event_index,
                "event_start_frame": case.event.start_frame,
                "event_end_frame": case.event.end_frame,
                "event_duration_sec": case.event.duration_sec,
                "duration_bin": case.duration_bin,
                "selection_reason": case.reason,
                "selection_rank_within_random_permutation": case.random_rank,
                "case_role": case.case_role, "video_standard_auc": case.video_standard_auc,
                "auc_rank": case.auc_rank,
            }
            for case in selected
        ],
    }
    if args.selection_mode == "high-auc-pool":
        selection["metrics_path"] = str(metrics_path)
        selection.update(audit)
    if args.dry_run:
        print(json.dumps(selection, indent=2))
        return selected
    commit = git_commit()
    worktree_dirty = git_worktree_dirty()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection["git_commit"] = commit
    selection["git_worktree_dirty"] = worktree_dirty
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if candidate_rows is not None:
        write_csv(args.output_dir / "candidate_videos.csv", CANDIDATE_VIDEO_FIELDS, candidate_rows)
    index = []
    for case in selected:
        index.append(export_case(
            case, infer_dir=args.infer_dir, video_root=args.video_root,
            gt_root=args.gt_root, manifest=manifest, output_dir=args.output_dir,
            seed=args.seed, score_mode=args.score_mode, pre_sec=args.pre_sec,
            post_sec=args.post_sec, frames_per_case=args.frames_per_case,
            dpi=args.dpi, commit=commit, worktree_dirty=worktree_dirty,
            plot_range=args.plot_range, export_local_zoom=args.export_local_zoom,
        ))
    write_csv(args.output_dir / "index.csv", INDEX_FIELDS, index)
    write_gallery(args.output_dir / "gallery.html", index)
    print(f"Exported {len(index)} GT-selected cases to {args.output_dir}")
    return selected


if __name__ == "__main__":
    main()
