import json

import pytest

from hivau_dataset import HIVAUDataset
from prepare_hivau_clip_labels import prepare_annotations


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _touch_video(root, video_id):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{video_id}.mp4").write_bytes(b"")


def test_normal_video_clip_labels_and_zero_window_soft_labels(tmp_path):
    raw = {
        "Normal_Videos001_x264": {
            "n_frames": 60,
            "fps": 10.0,
            "label": [],
            "events": [[1.0, 4.0]],
            "clips": [[[1.0, 2.0], [2.0, 3.0]]],
            "clips_caption": [[["person walks", "person stands"][0], "person stands"]],
        }
    }
    labeled, unresolved = prepare_annotations(raw)
    meta = labeled["Normal_Videos001_x264"]
    assert meta["clip_anomaly_labels"] == [[0, 0]]
    assert unresolved == []

    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, labeled)
    _touch_video(videos, "Normal_Videos001_x264")
    ds = HIVAUDataset(ann, videos, total_sampled_frames=4, sample_interval=5, max_windows=4)
    assert all(v == pytest.approx(0.0, abs=1e-7) for v in ds.samples[0]["clip_soft"])


def test_normal_event_in_abnormal_video_gets_zero_labels():
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 4.0]],
            "clips": [[[1.0, 2.0], [2.0, 3.0]]],
            "clips_caption": [["reading", "standing"]],
            "events_summary_split": [
                {"judgement": "No anomaly exists in this video."},
            ],
        }
    }
    labeled, unresolved = prepare_annotations(raw)
    assert labeled["Abuse001_x264"]["clip_anomaly_labels"] == [[0, 0]]
    assert unresolved == []


def test_unresolved_abnormal_clips_are_null_and_dataset_rejects_them(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 4.0]],
            "clips": [[[1.0, 2.0], [2.0, 3.0]]],
            "clips_caption": [["reading", "punching"]],
            "events_summary_split": [
                {"judgement": "An anomaly exists, specifically, Physical Abuse."},
            ],
        }
    }
    labeled, unresolved = prepare_annotations(raw)
    assert labeled["Abuse001_x264"]["clip_anomaly_labels"] == [[None, None]]
    assert len(unresolved) == 2

    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, labeled)
    _touch_video(videos, "Abuse001_x264")
    with pytest.raises(ValueError, match="Missing clip anomaly label: video=Abuse001_x264, event=0, clip=0"):
        HIVAUDataset(ann, videos, total_sampled_frames=4, sample_interval=5, max_windows=4)


def test_manual_labels_override_unresolved_abnormal_event():
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 4.0]],
            "clips": [[[1.0, 2.0], [2.0, 3.0]]],
            "clips_caption": [["reading", "punching"]],
            "events_summary_split": [
                {"judgement": "An anomaly exists, specifically, Physical Abuse."},
            ],
        }
    }
    labeled, unresolved = prepare_annotations(
        raw,
        {"Abuse001_x264": {"event0_clip0": 0, "event0_clip1": 1}},
    )
    assert labeled["Abuse001_x264"]["clip_anomaly_labels"] == [[0, 1]]
    assert unresolved == []


def test_overlapping_anomaly_clips_use_union_not_duplicate_counting(tmp_path):
    labeled = {
        "Abuse001_x264": {
            "n_frames": 50,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[0.0, 5.0]],
            "clips": [[[1.0, 3.0], [2.0, 4.0]]],
            "clips_caption": [["assault A", "assault B"]],
            "clip_anomaly_labels": [[1, 1]],
        }
    }
    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, labeled)
    _touch_video(videos, "Abuse001_x264")
    ds = HIVAUDataset(ann, videos, total_sampled_frames=4, sample_interval=5, max_windows=4)
    assert ds.samples[0]["clip_soft"][:2] == pytest.approx([0.5, 1.0], abs=1e-7)


def test_mixed_normal_and_abnormal_clips_only_count_abnormal_interval(tmp_path):
    labeled = {
        "Abuse001_x264": {
            "n_frames": 60,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[0.0, 6.0]],
            "clips": [[[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]],
            "clips_caption": [["normal start", "theft", "normal end"]],
            "clip_anomaly_labels": [[0, 1, 0]],
        }
    }
    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, labeled)
    _touch_video(videos, "Abuse001_x264")
    ds = HIVAUDataset(ann, videos, total_sampled_frames=6, sample_interval=10, max_windows=4)
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(2 / 6, abs=1e-7)


def test_summary_triggers_are_independent_of_anomaly_label(tmp_path):
    labeled = {
        "Normal_Videos001_x264": {
            "n_frames": 50,
            "fps": 10.0,
            "label": [],
            "events": [[1.0, 2.0]],
            "clips": [[[1.0, 2.0]]],
            "clips_caption": [["normal clip summary"]],
            "clip_anomaly_labels": [[0]],
        }
    }
    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, labeled)
    _touch_video(videos, "Normal_Videos001_x264")
    ds = HIVAUDataset(ann, videos, total_sampled_frames=4, sample_interval=5, max_windows=4)
    triggers = [t for window in ds.samples[0]["summary_triggers"] for t in window]
    assert len(triggers) == 1
    assert triggers[0]["text"] == "normal clip summary"
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(0.0, abs=1e-7)


def test_dataset_rejects_raw_events_without_clip_anomaly_labels(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 4.0]],
            "clips": [[[1.0, 2.0]]],
            "clips_caption": [["caption"]],
        }
    }
    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, raw)
    _touch_video(videos, "Abuse001_x264")
    with pytest.raises(ValueError, match="Run prepare_hivau_clip_labels.py first"):
        HIVAUDataset(ann, videos, total_sampled_frames=4, sample_interval=5, max_windows=4)
