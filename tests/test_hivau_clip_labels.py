import json
import math

import pytest

from hivau_dataset import HIVAUDataset, parse_event_judgement, seconds_to_frame_interval


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _touch_video(root, video_id):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{video_id}.mp4").write_bytes(b"")


def _make_dataset(
    tmp_path,
    raw,
    *,
    abnormal_ids=(),
    total_sampled_frames=4,
    sample_interval=5,
    max_windows=4,
):
    ann = tmp_path / "ann.json"
    videos = tmp_path / "videos"
    anomaly_root = tmp_path / "Anomaly-Videos-ALL"
    _write_json(ann, raw)
    for video_id in raw:
        _touch_video(videos, video_id)
    for video_id in abnormal_ids:
        _touch_video(anomaly_root / "Class", video_id)
    return HIVAUDataset(
        ann,
        videos,
        total_sampled_frames=total_sampled_frames,
        sample_interval=sample_interval,
        max_windows=max_windows,
        anomaly_video_root=anomaly_root if abnormal_ids else None,
    )


def test_normal_video_events_do_not_create_score_labels(tmp_path):
    raw = {
        "Normal_Videos001_x264": {
            "n_frames": 60,
            "fps": 10.0,
            "label": [],
            "events": [[1.0, 4.0]],
            "events_summary_split": [
                {"judgement": "The anomaly exists, but this text must not affect SCORE."},
            ],
            "clips": [[[1.0, 2.0], [2.0, 3.0]]],
            "clips_caption": [["person walks", "person stands"]],
        }
    }
    ds = _make_dataset(tmp_path, raw, abnormal_ids=())
    assert all(v == pytest.approx(0.0, abs=1e-7) for v in ds.samples[0]["clip_soft"])


def test_abnormal_video_filters_normal_events_by_judgement(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 3.0], [5.0, 7.0]],
            "events_summary_split": [
                {"judgement": "No anomaly exists in this video."},
                {"judgement": "This text is intentionally unparseable."},
            ],
            "clips": [[[1.0, 2.0]], [[5.0, 6.0]]],
            "clips_caption": [["normal action"], ["abnormal later action"]],
        }
    }
    ds = _make_dataset(tmp_path, raw, abnormal_ids=("Abuse001_x264",))
    # event 0 judged normal → excluded; event 1 unparseable → falls back to
    # video membership → kept.  Only windows covering [5.0, 7.0]s get labels.
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.0, 0.0, 0.5, 0.5], abs=1e-7)


def test_template_broken_abnormal_video_is_skipped(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 80,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 3.0], [5.0, 7.0]],
            "events_summary_split": [
                {"judgement": "No anomaly exists in this video."},
                {"judgement": "No anomaly exists here either."},
            ],
            "clips": [[[1.0, 2.0]], [[5.0, 6.0]]],
            "clips_caption": [["normal action"], ["normal action 2"]],
        }
    }
    ds = _make_dataset(tmp_path, raw, abnormal_ids=("Abuse001_x264",))
    # every event judged normal in an abnormal video → annotation template
    # error → the video is excluded instead of trained as fully normal
    assert len(ds.samples) == 0


def test_training_video_not_in_anomaly_root_is_normal_even_if_annotation_label_abuse(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 40,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 3.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    ds = _make_dataset(tmp_path, raw, abnormal_ids=("OtherAbuseVideo",))
    assert ds.samples[0]["video_label"] == 0
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.0, 0.0], abs=1e-7)


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
    ds = _make_dataset(
        tmp_path,
        raw,
        abnormal_ids=("Abuse001_x264",),
        total_sampled_frames=4,
        sample_interval=3,
        max_windows=2,
    )
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
    ds = _make_dataset(
        tmp_path,
        raw,
        abnormal_ids=("Abuse001_x264",),
        total_sampled_frames=6,
        sample_interval=10,
        max_windows=2,
    )
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(5 / 6, abs=1e-7)


def test_judgement_text_does_not_affect_score_or_raise(tmp_path):
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
    ds = _make_dataset(tmp_path, raw)
    assert all(v == pytest.approx(0.0, abs=1e-7) for v in ds.samples[0]["clip_soft"])


def test_unparseable_judgement_does_not_affect_abnormal_score_labels(tmp_path):
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
    ds = _make_dataset(tmp_path, raw, abnormal_ids=("Abuse001_x264",))
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.5, 1.0, 0.0], abs=1e-7)


def test_judgement_negation_has_priority():
    assert parse_event_judgement("No anomaly exists in this video.") == 0
    assert parse_event_judgement("An anomaly exists, specifically, Physical Abuse.") == 1


def test_judgement_phrasing_variants():
    # observed phrasings from the HIVAU UCF training annotations
    assert parse_event_judgement("No, the anomaly does not exist.") == 0
    assert parse_event_judgement("No anomaly exists, specifically no Abuse anomaly.") == 0
    assert parse_event_judgement("There exists an anomaly, specifically Arson.") == 1
    assert parse_event_judgement("There is an anomaly event (Arrest) in the video.") == 1
    assert parse_event_judgement("There is a suspected anomaly, specifically Shoplifting.") == 1
    assert parse_event_judgement("The anomaly exists and the man runs away.") == 1
    assert parse_event_judgement("There is a potential anomaly, specifically a Burglary.") == 1
    assert parse_event_judgement("There is a suspected anomaly, specifically Child Abuse.") == 1
    with pytest.raises(ValueError):
        parse_event_judgement("The scene contains people.")


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


def test_event_conversion_uses_each_video_fps(tmp_path):
    raw = {
        "Abuse10fps": {
            "n_frames": 40,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.0, 2.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        },
        "Abuse20fps": {
            "n_frames": 80,
            "fps": 20.0,
            "label": ["Abuse"],
            "events": [[1.0, 2.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        },
    }
    ds = _make_dataset(
        tmp_path,
        raw,
        abnormal_ids=("Abuse10fps", "Abuse20fps"),
        total_sampled_frames=4,
        sample_interval=5,
        max_windows=4,
    )
    by_video = {sample["video_id"]: sample["clip_soft"] for sample in ds.samples}
    assert by_video["Abuse10fps"] == pytest.approx([0.5, 0.0], abs=1e-7)
    assert by_video["Abuse20fps"] == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-7)


def test_half_open_event_boundary_on_sampled_frames(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 20,
            "fps": 1.0,
            "label": ["Abuse"],
            "events": [[3.0, 9.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    ds = _make_dataset(
        tmp_path,
        raw,
        abnormal_ids=("Abuse001_x264",),
        total_sampled_frames=4,
        sample_interval=3,
        max_windows=2,
    )
    # sampled frames in window 0 are 0, 3, 6, 9; half-open [3, 9) includes 3 and 6 only.
    assert ds.samples[0]["clip_soft"][0] == pytest.approx(0.5, abs=1e-7)


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
    ds = _make_dataset(
        tmp_path,
        raw,
        abnormal_ids=("Abuse001_x264",),
        total_sampled_frames=4,
        sample_interval=3,
        max_windows=2,
    )
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.0, 1.0], abs=1e-7)


def test_event_exceeding_video_end_is_clipped_with_warning(tmp_path):
    raw = {
        "Abuse001_x264": {
            "n_frames": 14,
            "fps": 10.0,
            "label": ["Abuse"],
            "events": [[1.2, 2.0]],
            "events_summary_split": [
                {"judgement": "An anomaly exists."},
            ],
            "clips": [[]],
            "clips_caption": [[]],
        }
    }
    with pytest.warns(RuntimeWarning, match="event time exceeds video bounds"):
        ds = _make_dataset(
            tmp_path,
            raw,
            abnormal_ids=("Abuse001_x264",),
            total_sampled_frames=4,
            sample_interval=3,
            max_windows=2,
        )
    assert ds.samples[0]["clip_soft"] == pytest.approx([0.0, 1.0], abs=1e-7)


def test_invalid_event_times_raise_clear_errors(tmp_path):
    bad_events = [
        [-0.1, 1.0],
        [1.0, 1.0],
        [float("nan"), 2.0],
    ]
    for idx, event in enumerate(bad_events):
        raw = {
            f"AbuseBad{idx}": {
                "n_frames": 40,
                "fps": 10.0,
                "label": ["Abuse"],
                "events": [event],
                "events_summary_split": [
                    {"judgement": "An anomaly exists."},
                ],
                "clips": [[]],
                "clips_caption": [[]],
            }
        }
        with pytest.raises(ValueError, match="Invalid time interval values"):
            _make_dataset(tmp_path / str(idx), raw, abnormal_ids=(f"AbuseBad{idx}",))


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
