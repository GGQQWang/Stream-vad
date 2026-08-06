"""Deprecated helper for explicit clip-level anomaly labels.

Current Stage 1 training does not require clip-level anomaly labels.  SCORE
supervision is built from ``events`` plus ``events_summary_split[].judgement``;
HIVAU ``clips`` and ``clips_caption`` are used only for SUMMARY supervision.
This script is kept for historical/manual-label experiments and is not called
by the Stage 1 data pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


NORMAL_JUDGEMENT_PHRASES = (
    "no anomaly exists",
    "there is no anomaly",
    "no anomaly exists in this video",
)


def _is_normal_video_label(label: Any) -> bool:
    if label is None:
        return False
    if isinstance(label, (list, tuple)):
        if len(label) == 0:
            return True
        return all(_is_normal_video_label(x) for x in label)
    if isinstance(label, str):
        return label.strip().lower() in {"normal", "0", "negative"}
    try:
        return int(label) == 0
    except (TypeError, ValueError):
        return False


def _is_normal_judgement(text: Any) -> bool:
    value = str(text or "").strip().lower()
    return any(phrase in value for phrase in NORMAL_JUDGEMENT_PHRASES)


def _manual_label_for(manual: dict, video_id: str, clip_id: str):
    value = manual.get(video_id, {}).get(clip_id)
    if value is None:
        return None
    if value not in (0, 1):
        raise ValueError(f"manual label must be 0 or 1: video={video_id}, clip_id={clip_id}, value={value!r}")
    return int(value)


def label_video_clips(video_id: str, meta: dict, manual_labels: dict) -> tuple[list, list, list]:
    clips = meta.get("clips", [])
    captions = meta.get("clips_caption", [])
    summaries = meta.get("events_summary_split", [])
    if len(clips) != len(captions):
        raise ValueError(
            f"clips/clips_caption event count mismatch: video={video_id}, "
            f"clips={len(clips)}, captions={len(captions)}"
        )

    is_normal_video = _is_normal_video_label(meta.get("label"))
    clip_anomaly_labels: List[List[int | None]] = []
    clip_annotations: List[List[dict]] = []
    unresolved: List[dict] = []

    for event_idx, event_clips in enumerate(clips):
        event_captions = captions[event_idx]
        if len(event_clips) != len(event_captions):
            raise ValueError(
                f"clips/clips_caption clip count mismatch: "
                f"video={video_id}, event={event_idx}, "
                f"clips={len(event_clips)}, captions={len(event_captions)}"
            )
        judgement = ""
        if event_idx < len(summaries) and isinstance(summaries[event_idx], dict):
            judgement = str(summaries[event_idx].get("judgement", ""))
        is_normal_event = _is_normal_judgement(judgement)

        labels_row: List[int | None] = []
        annotations_row: List[dict] = []
        for clip_idx, (clip_range, caption) in enumerate(zip(event_clips, event_captions)):
            clip_id = f"event{event_idx}_clip{clip_idx}"
            manual_value = _manual_label_for(manual_labels, video_id, clip_id)
            if manual_value is not None:
                anomaly_label = manual_value
                source = "manual"
            elif is_normal_video:
                anomaly_label = 0
                source = "normal_video"
            elif is_normal_event:
                anomaly_label = 0
                source = "normal_event_judgement"
            else:
                anomaly_label = None
                source = "unresolved"

            start, end = clip_range
            item = {
                "clip_id": clip_id,
                "start": float(start),
                "end": float(end),
                "caption": str(caption),
                "anomaly_label": anomaly_label,
                "label_source": source,
            }
            labels_row.append(anomaly_label)
            annotations_row.append(item)
            if anomaly_label is None:
                unresolved.append({
                    "video_id": video_id,
                    "video_label": meta.get("label"),
                    "event_index": event_idx,
                    "clip_index": clip_idx,
                    "clip_id": clip_id,
                    "start": float(start),
                    "end": float(end),
                    "caption": str(caption),
                    "event_judgement": judgement,
                })

        clip_anomaly_labels.append(labels_row)
        clip_annotations.append(annotations_row)

    return clip_anomaly_labels, clip_annotations, unresolved


def prepare_annotations(raw: dict, manual_labels: dict | None = None) -> tuple[dict, list]:
    manual = manual_labels or {}
    output = {}
    unresolved_all = []
    for video_id, meta in raw.items():
        new_meta = dict(meta)
        labels, annotations, unresolved = label_video_clips(video_id, meta, manual)
        new_meta["clip_anomaly_labels"] = labels
        new_meta["clip_annotations"] = annotations
        output[video_id] = new_meta
        unresolved_all.extend(unresolved)
    return output, unresolved_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--manual-label-json", default="")
    parser.add_argument("--unresolved-output", default="hivau_unresolved_clips.json")
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        raw = json.load(f)
    manual = {}
    if args.manual_label_json:
        with open(args.manual_label_json, "r") as f:
            manual = json.load(f)

    labeled, unresolved = prepare_annotations(raw, manual)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)

    unresolved_path = Path(args.unresolved_output)
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(unresolved_path, "w") as f:
        json.dump(unresolved, f, indent=2, ensure_ascii=False)

    print(
        f"Wrote labeled annotations to {output_path}; "
        f"unresolved_clips={len(unresolved)} -> {unresolved_path}"
    )


if __name__ == "__main__":
    main()
