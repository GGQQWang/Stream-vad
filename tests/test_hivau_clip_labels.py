import json
import math

import pytest

from hivau_dataset import HIVAUDataset, parse_event_judgement, seconds_to_frame_interval


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _touch_video(root, video_id):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{video_id}.mp4").write_bytes(b"")


def _make_dataset(tmp_path, raw, *, total_sampled_frames=4, sample_interval=5, max_windows=4):
    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    _write_json(ann, raw)
    for video_id in raw:
        _touch_video(videos, video_id)
    return HIVAUDataset(
        ann,
        videos,
        total_sampled_frames=total_sampled_frames,
        sample_interval=sample_interval,
        max_windows=max_windows,
    )


def test_normal_video_events_do_not_create_score_labels(tmp_path):
    raw = {
        "Normal_Videos001_x264": {
            "n_frames": 60,
            "fps": 10.0,
            "label": [],
            "events": [[1.0, 4.0]],
            "events_summary_split": [
                {"judgement": "There is no anomaly in the video."},
            ],
            "clips": [[[1.0, 2.0], [2.0, 3.0]]],
            "clips_caption": [["person walks", "person stands"]],
        }
    }
    ds = _make_dataset(tmp_path, raw)
    assert all(v == pytest.approx(0.0, abs=1e-7) for v in ds.samples[0]["clip_soft"])


def test_abnormal_video_uses_only_abnormal_event_for_score(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 3.0], [5.0, 7.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists, specifically, Physical Abuse."},
                {"judgement": "No anomaly exists in this video."},
            ],
            "clips": [[[1.0, 2.0]], [[5.0, 6.0]]],
            "clips_caption": [["abnormal action"], ["normal later action"]],
        }
    }
    ds = _make_dataset(tmp_path, raw)
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.5, 0.5, 0.0, 0.0], abs=1e-7)


def test_partial_window_overlap_uses_sampled_frames(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 12,
            "fps": 1.0,
            "label": ["Abuse"],
            "events": [[6.0, 10.0]],
            "events_summary_split": [
                {"judgement": "An abnormal event exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    ds = _make_dataset(tmp_path, raw, total_sampled_frames=4, sample_interval=3, max_windows=2)
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(0.5, abs=1e-7)


def test_overlapping_abnormal_events_use_union_not_duplicate_counting(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 70,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 4.0], [3.0, 6.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[], []],
            "clips_caption": [[], []],
        }
    }
    ds = _make_dataset(tmp_path, raw, total_sampled_frames=6, sample_interval=10, max_windows=2)
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(5 / 6, abs=1e-7)


def test_normal_video_with_anomaly_judgement_is_conflict(tmp_path):
    raw = {
        "Normal_Videos001_x264": {
            "n_frames": 60,
            "fps": 10.0,
            "label": [],
            "events": [[1.0, 4.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists, specifically, Physical Abuse."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    with pytest.raises(ValueError, match="video label says normal, but event judgement says anomaly"):
        _make_dataset(tmp_path, raw)


def test_unparseable_judgement_raises_with_video_and_event_idx(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 60,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 4.0]],
            "events_summary_split": [
                {"judgement": "The scene contains people."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    with pytest.raises(ValueError, match="video=Abuse001_x264, event_idx=0"):
        _make_dataset(tmp_path, raw)


def test_judgement_negation_has_priority():
    assert parse_event_judgement("No anomaly exists in this video.") == 0
    assert parse_event_judgement("An anomaly exists, specifically, Physical Abuse.") == 1


def test_seconds_to_frame_interval_uses_floor_and_ceil():
    interval = seconds_to_frame_interval(
        3.933,
        15.867,
        fps=30.0,
        n_frames=1000,
        video_id="v",
        event_idx=0,
    )
    assert interval == (math.floor(3.933 * 30), math.ceil(15.867 * 30))


def test_tail_short_window_uses_valid_sampled_frame_denominator(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 14,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.2, 1.3]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    ds = _make_dataset(tmp_path, raw, total_sampled_frames=4, sample_interval=3, max_windows=2)
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.0, 1.0], abs=1e-7)


def test_summary_triggers_are_independent_of_event_score_label(tmp_path):
    raw = {
        "Normal_Videos001_x264": {
            "n_frames": 50,
            "fps": 10.0,
            "label": [],
            "events": [[1.0, 2.0]],
            "events_summary_split": [
                {"judgement": "No anomaly exists in this video."},
            ],
            "clips": [[[1.0, 2.0]]],
            "clips_caption": [["A person walks."]],
        }
    }
    ds = _make_dataset(tmp_path, raw)
    triggers = [t for window in ds.samples[0]["summary_triggers"] for t in window]
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(0.0, abs=1e-7)
    assert len(triggers) == 1
    assert triggers[0]["text"] == "A person walks."


def test_same_window_multiple_summary_triggers_are_preserved(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[0.0, 8.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[[2.1, 3.1], [2.2, 3.2]]],
            "clips_caption": [["first caption", "second caption"]],
        }
    }
    ds = _make_dataset(tmp_path, raw)
    triggers = [t for window in ds.samples[0]["summary_triggers"] for t in window]
    assert [t["text"] for t in triggers] == ["first caption", "second caption"]
