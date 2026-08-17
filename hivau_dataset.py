"""HIVAU-70K dataset — video chunked into fixed-size windows for training.

Each video is pre-cut into non-overlapping, contiguous chunks of at most
``max_windows`` scoring windows.  A chunk is a memory/TBPTT unit, not a
semantic HIVAU clip.  HIVAU clip-level captions are represented as summary
triggers on the window whose time range contains the clip end.

Chunks carry ``valid_mask`` so the last (possibly shorter) chunk is handled
correctly in loss, metrics, and logging.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from feature_cache import load_feature_cache
from stage1_streaming import build_window_infos, sampled_fps, window_span_frames


def _read_video_label(meta: dict, frame_labels=None) -> int:
    """Prefer explicit video-level labels without using event locations."""
    if "video_label" in meta:
        return int(meta["video_label"])
    if "label" in meta:
        label = meta["label"]
        if isinstance(label, (list, tuple)):
            if len(label) == 0:
                return 0
            return 1 if any(_label_value_is_abnormal(x) for x in label) else 0
        return 1 if _label_value_is_abnormal(label) else 0
    return 0


def _label_value_is_abnormal(label) -> bool:
    if isinstance(label, str):
        value = label.strip().lower()
        return value not in {"normal", "0", "negative"}
    try:
        return int(label) != 0
    except (TypeError, ValueError):
        return True


def collect_abnormal_video_ids(anomaly_video_root: str | Path | None) -> set[str]:
    """Collect abnormal video ids from an Anomaly-Videos-ALL tree.

    The original UCF split defines abnormality by membership in the anomaly
    video collection.  File stems are used as video ids, e.g. ``Abuse001_x264``
    for ``Abuse001_x264.mp4``.
    """
    if not anomaly_video_root:
        return set()
    root = Path(anomaly_video_root)
    if not root.exists():
        raise FileNotFoundError(f"Anomaly video root does not exist: {root}")
    ids: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file():
            ids.add(path.stem)
    return ids


def parse_event_judgement(judgement: str, video_id: str = "", event_idx: int | None = None) -> int:
    """Parse HIVAU event judgement into 0=normal or 1=anomaly.

    Negated phrases are checked before positive phrases so strings such as
    "No anomaly exists in this video" are not misread as abnormal.
    """
    if not isinstance(judgement, str):
        raise ValueError(
            f"Unparseable event judgement: video={video_id}, "
            f"event_idx={event_idx}, judgement={judgement!r}"
        )
    normalized = " ".join(judgement.lower().strip().split())
    if not normalized:
        raise ValueError(
            f"Unparseable event judgement: video={video_id}, "
            f"event_idx={event_idx}, judgement={judgement!r}"
        )
    negative_patterns = (
        "there is no anomaly",
        "there are no anomalies",
        "no anomaly",
        "no anomalies",
        "no abnormal event",
        "no abnormal events",
        "no anomaly event",
        "no anomaly events",
        "anomaly does not exist",
        "anomalies do not exist",
        "anomaly is not present",
        "abnormal event does not exist",
    )
    positive_patterns = (
        "anomaly exists",
        "anomalies exist",
        "an anomaly",
        "anomalous event",
        "abnormal event",
        "potential anomaly",
        "suspected anomaly",
    )
    if any(pattern in normalized for pattern in negative_patterns):
        return 0
    if any(pattern in normalized for pattern in positive_patterns):
        return 1
    raise ValueError(
        f"Unparseable event judgement: video={video_id}, "
        f"event_idx={event_idx}, judgement={judgement!r}"
    )


def _validate_time_interval(
    interval: Sequence[float],
    *,
    video_id: str,
    field: str,
    event_idx: int,
    clip_idx: int | None = None,
) -> tuple[float, float]:
    loc = f"video={video_id}, {field}, event={event_idx}"
    if clip_idx is not None:
        loc += f", clip={clip_idx}"
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"Invalid time interval length: {loc}, interval={interval!r}")
    try:
        start = float(interval[0])
        end = float(interval[1])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid time interval values: {loc}, interval={interval!r}")
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or not (start < end):
        raise ValueError(f"Invalid time interval values: {loc}, interval={interval!r}")
    return start, end


def seconds_to_frame_interval(
    start_sec: float,
    end_sec: float,
    *,
    fps: float,
    n_frames: int,
    video_id: str = "",
    event_idx: int | None = None,
    field: str = "event",
) -> tuple[int, int] | None:
    if fps <= 0 or not math.isfinite(float(fps)):
        raise ValueError(f"Invalid fps: video={video_id}, fps={fps!r}")
    if n_frames <= 0:
        raise ValueError(f"Invalid n_frames: video={video_id}, n_frames={n_frames!r}")
    start_frame = math.floor(float(start_sec) * float(fps))
    end_frame = math.ceil(float(end_sec) * float(fps))
    clipped_start = max(0, start_frame)
    clipped_end = min(int(n_frames), end_frame)
    if clipped_start != start_frame or clipped_end != end_frame:
        warnings.warn(
            f"{field} time exceeds video bounds and was clipped: "
            f"video={video_id}, event_idx={event_idx}, "
            f"seconds=({start_sec}, {end_sec}), frames=({start_frame}, {end_frame}), "
            f"n_frames={n_frames}",
            RuntimeWarning,
        )
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _extract_event_intervals_for_score(
    meta: dict,
    video_id: str,
    fps: float,
    n_frames: int,
    is_abnormal_video: bool,
    debug_events: bool = False,
) -> tuple[List[List[float]], int, int, bool]:
    """Extract per-event abnormal intervals for SCORE supervision.

    Abnormal videos may contain normal events: HIVAU annotates every
    temporal event in a video and stores the per-event verdict in
    ``events_summary_split[i]["judgement"]`` ("An anomaly exists" /
    "No anomaly exists").  Events whose judgement is normal are excluded
    even when the video itself is abnormal.  Events without a judgement
    fall back to video-level membership.

    Returns ``(intervals, abnormal_events, normal_events, skip_video)``.
    ``skip_video`` is True only for abnormal videos whose event judgements
    leave zero abnormal events (annotation template errors); such videos
    must be excluded from training entirely.
    """
    events = meta.get("events", []) or []
    event_judgements = meta.get("events_summary_split", []) or []

    intervals: List[List[float]] = []
    abnormal_events = 0
    normal_events = 0
    for event_idx, event_range in enumerate(events):
        start_sec, end_sec = _validate_time_interval(
            event_range,
            video_id=video_id,
            field="events",
            event_idx=event_idx,
        )
        frame_interval = seconds_to_frame_interval(
            start_sec,
            end_sec,
            fps=fps,
            n_frames=n_frames,
            video_id=video_id,
            event_idx=event_idx,
            field="event",
        )

        # per-event anomaly decision: prefer the judgement text over
        # video-level membership so normal events inside abnormal videos
        # are not mislabeled as abnormal
        is_abnormal_event = False
        decision_source = "video_label"
        if is_abnormal_video:
            is_abnormal_event = True
            if event_idx < len(event_judgements) and isinstance(event_judgements[event_idx], dict):
                judgement = str(event_judgements[event_idx].get("judgement", ""))
                if judgement:
                    try:
                        is_abnormal_event = bool(parse_event_judgement(
                            judgement, video_id=video_id, event_idx=event_idx,
                        ))
                        decision_source = "event_judgement"
                    except ValueError:
                        warnings.warn(
                            f"unparseable event judgement; falling back to video "
                            f"membership: video={video_id}, event_idx={event_idx}, "
                            f"judgement={judgement!r}",
                            RuntimeWarning,
                        )
                        decision_source = "video_label_fallback"

        if debug_events:
            print(
                f"HIVAU event: video={video_id}, event_idx={event_idx}, "
                f"seconds=({start_sec}, {end_sec}), frames={frame_interval}, "
                f"video_label={int(is_abnormal_video)}, "
                f"abnormal={int(is_abnormal_event)} ({decision_source})"
            )
        if is_abnormal_event:
            abnormal_events += 1
            intervals.append([start_sec, end_sec])
        else:
            normal_events += 1

    skip_video = bool(is_abnormal_video and abnormal_events == 0)
    return intervals, abnormal_events, normal_events, skip_video


def _pad_int(values: List[int], length: int, pad_value: int) -> List[int]:
    out = list(values)
    if len(out) < length:
        out.extend([pad_value] * (length - len(out)))
    return out


def _pad_list(values: List[list], length: int) -> List[list]:
    out = list(values)
    if len(out) < length:
        out.extend([[] for _ in range(length - len(out))])
    return out


def _parse_summary_clips(meta: dict, fps: float, n_frames: int, video_id: str = "") -> List[dict]:
    """Parse HIVAU clip-level text annotations into frame boundaries.

    Supported formats:
      - ``summary_clips``: list of dicts with frame or second boundaries.
      - ``clips`` + ``clips_caption``: nested HIVAU-style second intervals.

    Clip endings are handled later by ``build_window_infos`` and trigger on
    the window whose time range contains the clip end frame.
    """
    parsed: List[dict] = []
    seen_summary: set[tuple[int, int, str]] = set()

    def _append_summary(clip_id: str, start: int, end: int, text) -> None:
        caption = str(text)
        if not caption:
            return
        key = (max(0, start), min(n_frames, end), caption)
        if key[1] <= key[0]:
            return
        if key in seen_summary:
            return
        seen_summary.add(key)
        parsed.append({
            "clip_id": clip_id,
            "clip_start_frame": key[0],
            "clip_end_frame": key[1],
            "text": caption,
        })

    if isinstance(meta.get("summary_clips"), list):
        for i, item in enumerate(meta["summary_clips"]):
            if not isinstance(item, dict):
                continue
            if "clip_start_frame" in item:
                start = int(item["clip_start_frame"])
            else:
                start = math.floor(float(item.get("clip_start", item.get("start", 0.0))) * fps)
            if "clip_end_frame" in item:
                end = int(item["clip_end_frame"])
            else:
                end = math.ceil(float(item.get("clip_end", item.get("end", 0.0))) * fps)
            _append_summary(item.get("clip_id", f"summary_{i}"), start, end, item.get("text", item.get("caption", "")))

    raw_clips = meta.get("clips", None)
    raw_captions = meta.get("clips_caption", None)
    if raw_clips is not None and raw_captions is not None:
        if len(raw_clips) != len(raw_captions):
            raise ValueError(
                f"clips/clips_caption event count mismatch: video={video_id}, "
                f"clips={len(raw_clips)}, captions={len(raw_captions)}"
            )
        events = meta.get("events", []) or []
        for event_idx, event_clips in enumerate(raw_clips):
            event_captions = raw_captions[event_idx]
            if len(event_clips) != len(event_captions):
                raise ValueError(
                    f"clips/clips_caption clip count mismatch: "
                    f"video={video_id}, event={event_idx}, "
                    f"clips={len(event_clips)}, captions={len(event_captions)}"
                )
            event_frame_interval = None
            if event_idx < len(events):
                es, ee = _validate_time_interval(
                    events[event_idx],
                    video_id=video_id,
                    field="events",
                    event_idx=event_idx,
                )
                event_frame_interval = seconds_to_frame_interval(
                    es,
                    ee,
                    fps=fps,
                    n_frames=n_frames,
                    video_id=video_id,
                    event_idx=event_idx,
                    field="event",
                )
            for clip_idx, (clip_range, cap) in enumerate(zip(event_clips, event_captions)):
                cs, ce = _validate_time_interval(
                    clip_range,
                    video_id=video_id,
                    field="clips",
                    event_idx=event_idx,
                    clip_idx=clip_idx,
                )
                clip_frame_interval = seconds_to_frame_interval(
                    cs,
                    ce,
                    fps=fps,
                    n_frames=n_frames,
                    video_id=video_id,
                    event_idx=event_idx,
                    field="clip",
                )
                if clip_frame_interval is None:
                    continue
                if event_frame_interval is not None:
                    ev_start, ev_end = event_frame_interval
                    cl_start, cl_end = clip_frame_interval
                    if cl_start < ev_start or cl_end > ev_end:
                        warnings.warn(
                            f"clip time is outside matching event bounds: "
                            f"video={video_id}, event_idx={event_idx}, clip_idx={clip_idx}, "
                            f"clip_frames=({cl_start}, {cl_end}), event_frames=({ev_start}, {ev_end})",
                            RuntimeWarning,
                        )
                _append_summary(
                    f"event{event_idx}_clip{clip_idx}",
                    clip_frame_interval[0],
                    clip_frame_interval[1],
                    cap,
                )
    return parsed


class HIVAUDataset(Dataset):
    """HIVAU-70K — per-chunk window sequences.

    Each ``__getitem__`` returns a fixed-size chunk of ``max_windows``
    consecutive clips (the last chunk may be shorter, padded with
    ``valid_mask == 0``).

    Args:
        annotation_path: ``*_database_*.json``.
        video_root: directory containing .mp4 files.
        total_sampled_frames: sampled frames per scoring window.  Default 16.
        sample_interval: source-frame stride inside a scoring window.  Default 3.
        max_windows: max scoring windows per chunk.  Default 8.
        fps: fallback frame rate.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        video_root: str | Path,
        total_sampled_frames: int = 16,
        sample_interval: int = 3,
        max_windows: int = 8,
        fps: float = 30.0,
        feature_cache_root: str | Path | None = None,
        feature_cache_model_id: str = "",
        min_pixels: int = 200704,
        max_pixels: int = 200704,
        debug_events: bool = False,
        anomaly_video_root: str | Path | None = None,
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
        self.use_abnormal_video_ids = anomaly_video_root is not None and str(anomaly_video_root) != ""
        self.abnormal_video_ids = collect_abnormal_video_ids(anomaly_video_root)

        # ---- load annotations ----
        with open(annotation_path, "r") as f:
            raw = json.load(f)

        # ---- pre-compute per-video clip labels, then chunk ----
        self.samples: List[dict] = []
        total_windows_all = 0
        video_count = 0
        normal_video_count = 0
        abnormal_video_count = 0
        total_event_count = 0
        abnormal_event_count = 0
        normal_event_count = 0
        total_summary_triggers = 0
        skipped_template_broken = 0

        for video_name, meta in raw.items():
            n = meta["n_frames"]
            if n <= 0:
                continue
            video_fps = meta.get("fps", fps)
            video_path = self.video_root / f"{video_name}.mp4"
            if self.feature_cache_root is None and not video_path.exists():
                continue

            if self.use_abnormal_video_ids:
                video_label = 1 if video_name in self.abnormal_video_ids else 0
            else:
                video_label = _read_video_label(meta)
            if video_label == 0:
                normal_video_count += 1
            else:
                abnormal_video_count += 1
            video_count += 1
            anomaly_event_intervals, n_abnormal_events, n_normal_events, skip_video = _extract_event_intervals_for_score(
                meta,
                video_name,
                video_fps,
                n,
                is_abnormal_video=(video_label == 1),
                debug_events=debug_events,
            )
            if skip_video:
                # abnormal video whose every event judgement says "no anomaly"
                # (annotation template error): exclude entirely rather than
                # training it as a fully normal sample
                skipped_template_broken += 1
                if debug_events:
                    print(f"SKIP template-broken abnormal video: {video_name}")
                continue
            abnormal_event_count += n_abnormal_events
            normal_event_count += n_normal_events
            total_event_count += n_abnormal_events + n_normal_events
            summary_clips = _parse_summary_clips(meta, video_fps, n, video_name)
            # ``window`` is the scoring time step.  It is not the HIVAU
            # semantic clip.  A chunk is only a TBPTT/memory management unit.
            window_infos = build_window_infos(
                n_frames=n,
                fps=video_fps,
                anomaly_intervals=anomaly_event_intervals,
                frames_per_clip=total_sampled_frames,
                sample_interval=sample_interval,
                summary_clips=summary_clips,
            )
            clip_soft = [w.soft_label for w in window_infos]
            clip_bin = [1 if w.soft_label > 0.0 else 0 for w in window_infos]
            n_clips = math.ceil(n / self.clip_span)
            total_summary_triggers += sum(len(w.summary_triggers) for w in window_infos)

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
                    "sampled_fps": sampled_fps(video_fps, sample_interval),
                    "window_span_frames": window_span_frames(total_sampled_frames, sample_interval),
                    "chunk_start": lo,
                    "chunk_end": hi,
                    "n_total_windows": n_clips,
                    "is_last_chunk": (hi == n_clips),
                    "clip_soft": clip_soft[lo:hi],
                    "clip_bin": clip_bin[lo:hi],
                    "window_start_frames": [w.start_frame for w in window_infos[lo:hi]],
                    "window_end_frames": [w.end_frame for w in window_infos[lo:hi]],
                    "valid_end_frames": [w.valid_end_frame for w in window_infos[lo:hi]],
                    "summary_triggers": [list(w.summary_triggers) for w in window_infos[lo:hi]],
                    "skipped_summary_boundaries": [
                        w.skipped_summary_boundaries for w in window_infos[lo:hi]
                    ],
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
            f"{total_windows_all} total windows; "
            f"videos={video_count}, normal_videos={normal_video_count}, "
            f"abnormal_videos={abnormal_video_count}, events={total_event_count}, "
            f"abnormal_events={abnormal_event_count}, normal_events={normal_event_count}, "
            f"summary_triggers={total_summary_triggers}, "
            f"skipped_template_broken={skipped_template_broken}"
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
                "fps": meta["fps"],
                "chunk_start": meta["chunk_start"],
                "n_total_windows": meta["n_total_windows"],
                "is_last_chunk": meta["is_last_chunk"],
                "features": features,
                "labels": soft,
                "binary": binary,
                "valid_mask": valid,
                "window_start_frames": _pad_int(meta["window_start_frames"], self.max_windows, -1),
                "window_end_frames": _pad_int(meta["window_end_frames"], self.max_windows, -1),
                "valid_end_frames": _pad_int(meta["valid_end_frames"], self.max_windows, -1),
                "summary_triggers": _pad_list(meta["summary_triggers"], self.max_windows),
                "skipped_summary_boundaries": _pad_int(
                    meta["skipped_summary_boundaries"], self.max_windows, 0,
                ),
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
            "fps": meta["fps"],
            "chunk_start": meta["chunk_start"],
            "n_total_windows": meta["n_total_windows"],
            "is_last_chunk": meta["is_last_chunk"],
            "frames": frames,
            "labels": soft,
            "binary": binary,
            "valid_mask": valid,
            "window_start_frames": _pad_int(meta["window_start_frames"], self.max_windows, -1),
            "window_end_frames": _pad_int(meta["window_end_frames"], self.max_windows, -1),
            "valid_end_frames": _pad_int(meta["valid_end_frames"], self.max_windows, -1),
            "summary_triggers": _pad_list(meta["summary_triggers"], self.max_windows),
            "skipped_summary_boundaries": _pad_int(
                meta["skipped_summary_boundaries"], self.max_windows, 0,
            ),
        }

def hivau_collate(batch: List[dict]) -> dict:
    """Collate — frames stay as list (different resolutions safe)."""
    out = {
        "video_path": [b["video_path"] for b in batch],
        "video_id": [b["video_id"] for b in batch],
        "video_label": [b["video_label"] for b in batch],
        "fps": [b["fps"] for b in batch],
        "chunk_start": [b["chunk_start"] for b in batch],
        "n_total_windows": [b["n_total_windows"] for b in batch],
        "is_last_chunk": [b["is_last_chunk"] for b in batch],
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "binary": torch.stack([b["binary"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "window_start_frames": torch.tensor([b["window_start_frames"] for b in batch], dtype=torch.long),
        "window_end_frames": torch.tensor([b["window_end_frames"] for b in batch], dtype=torch.long),
        "valid_end_frames": torch.tensor([b["valid_end_frames"] for b in batch], dtype=torch.long),
        "summary_triggers": [b.get("summary_triggers", []) for b in batch],
        "skipped_summary_boundaries": [b.get("skipped_summary_boundaries", []) for b in batch],
    }
    if "features" in batch[0]:
        out["features"] = torch.stack([b["features"] for b in batch], dim=0)
    else:
        out["frames"] = [b["frames"] for b in batch]     # list of [max_w, F, C, H, W]
    return out
