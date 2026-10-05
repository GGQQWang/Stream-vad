"""Regression tests for GT-selected, causal qualitative exports."""

import csv
import json
import numpy as np
import pytest

from tools import export_case_studies as cases


def _event(video_id="video", index=0, start=10, end=20, fps=10.0, n_frames=50):
    return cases.Event(video_id, index, start, end, fps, n_frames)


def _write_gt(root, video_id, labels):
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(root / f"{video_id}.txt", labels, fmt="%d")


def _write_windows(path, video_id, scores, *, fps=10.0, span=10):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cases.WINDOW_FIELDS)
        writer.writeheader()
        for i, score in enumerate(scores):
            start, end = i * span, (i + 1) * span
            writer.writerow({
                "video_id": video_id, "window_index": i,
                "start_frame": start, "end_frame": end,
                "start_sec": start / fps, "end_sec": end / fps,
                "score_prob": score,
            })


def test_contiguous_gt_events_are_half_open_and_multiple():
    gt = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=int)
    assert cases.contiguous_events(gt) == [(0, 2), (3, 6), (8, 9)]
    assert cases.contiguous_events(np.zeros(5, dtype=int)) == []
    assert cases.contiguous_events(np.ones(5, dtype=int)) == [(0, 5)]


def test_manifest_keeps_internal_dots_and_rejects_collisions(tmp_path):
    ids = ["Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6",
           "Deadpool.2.2018__#0-04-46_0-05-01_label_B2-0-0"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({video_id: {"fps": 29.97, "n_frames": 90} for video_id in ids}))
    assert list(cases.load_manifest(manifest)) == ids
    assert cases.normalize_video_id(ids[0]) == ids[0]
    assert cases.normalize_video_id(f"{ids[0]}.MP4") == ids[0]
    manifest.write_text(json.dumps({"a": {"fps": 30, "n_frames": 5},
                                    "a.mp4": {"fps": 30, "n_frames": 5}}))
    with pytest.raises(ValueError, match="duplicate normalized"):
        cases.load_manifest(manifest)


def test_candidates_use_only_gt_and_skip_normal_videos(tmp_path):
    manifest = {
        "abnormal": {"fps": 29.97, "n_frames": 8},
        "normal": {"fps": 30.0, "n_frames": 8},
    }
    _write_gt(tmp_path, "abnormal", [0, 1, 1, 0, 0, 1, 0, 0])
    _write_gt(tmp_path, "normal", [0] * 8)
    events = cases.build_candidates(manifest, tmp_path)
    assert [(e.video_id, e.event_index, e.start_frame, e.end_frame) for e in events] == [
        ("abnormal", 0, 1, 3), ("abnormal", 1, 5, 6),
    ]
    assert events[0].case_id == "abnormal__event00"
    assert events[0].duration_sec == pytest.approx(2 / 29.97)


def test_duration_bins_use_global_tertiles():
    events = [_event(f"v{i}", end=i + 1) for i in range(1, 7)]
    bins, quantiles = cases.duration_bins(events)
    assert quantiles == pytest.approx(tuple(np.quantile([e.duration_sec for e in events], [1 / 3, 2 / 3])))
    assert [bins[e.case_id] for e in events] == ["short", "short", "medium", "medium", "long", "long"]


def test_stratified_selection_reproducible_and_records_shortages():
    events = [_event(f"s{i}", end=11) for i in range(2)]
    events += [_event(f"m{i}", end=31) for i in range(7)]
    events += [_event(f"l{i}", end=51, n_frames=70) for i in range(7)]
    bins = {e.case_id: "short" if e.video_id[0] == "s" else "medium" if e.video_id[0] == "m" else "long" for e in events}
    first, audit = cases.select_cases(events, bins, mode="stratified-random", num_cases=12, seed=42)
    second, _ = cases.select_cases(list(reversed(events)), bins, mode="stratified-random", num_cases=12, seed=42)
    assert [c.event.case_id for c in first] == [c.event.case_id for c in second]
    assert audit["initial_shortage"] == {"short": 2, "medium": 0, "long": 0}
    assert audit["allocated"] == {"short": 2, "medium": 5, "long": 5}
    assert len(first) == 12
    assert all(c.random_rank is not None for c in first)


def test_dry_run_selection_ignores_model_scores(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    gt_root = tmp_path / "gt"
    infer_dir = tmp_path / "inference"
    infer_dir.mkdir()
    manifest.write_text(json.dumps({f"v{i}": {"fps": 10, "n_frames": 20} for i in range(6)}))
    for i in range(6):
        _write_gt(gt_root, f"v{i}", [0] * i + [1] * (i + 1) + [0] * (19 - 2 * i))
    args = ["--infer-dir", str(infer_dir), "--video-root", str(tmp_path),
            "--gt-root", str(gt_root), "--manifest", str(manifest),
            "--output-dir", str(tmp_path / "out"), "--num-cases", "3", "--dry-run"]
    first = cases.main(args)
    capsys.readouterr()
    for i in range(6):
        _write_windows(infer_dir / f"v{i}_window_scores.csv", f"v{i}", [0.99, 0.01])
    second = cases.main(args)
    assert [c.event.case_id for c in first] == [c.event.case_id for c in second]
    assert not (tmp_path / "out").exists()


def test_temporal_anchors_short_event_deduplicates_and_clamps():
    event = _event(start=0, end=1, fps=29.97, n_frames=3)
    anchors, duplicates = cases.temporal_anchors(event, pre_sec=3, post_sec=3, frames_per_case=7)
    assert anchors == [("start", 0), ("post", 2)]
    assert len(duplicates) == 4
    assert all(d["frame_index"] == 0 for d in duplicates)
    assert cases.context_interval(event, pre_sec=3, post_sec=3) == pytest.approx((0, 3 / 29.97))


def test_default_anchors_depend_on_gt_time_not_score():
    event = _event(start=10, end=19, fps=10, n_frames=40)
    anchors, duplicates = cases.temporal_anchors(event, pre_sec=1, post_sec=1, frames_per_case=7)
    assert [label for label, _ in anchors] == ["pre", "start", "early", "middle", "late", "end", "post"]
    assert [frame for _, frame in anchors] == [5, 10, 12, 14, 16, 18, 23]
    assert duplicates == []


def test_window_end_alignment_uses_end_sec_without_shifting(tmp_path):
    event = _event(start=10, end=20, fps=10, n_frames=30)
    case = cases.SelectedCase(event, "medium", "manual", "manual")
    path = tmp_path / "video_window_scores.csv"
    _write_windows(path, "video", [0.1, 0.8, 0.3])
    rows = cases.read_window_scores(path, video_id="video", n_frames=30, fps=10)
    x, y, note = cases.score_points(tmp_path, case, rows, (0, 3), "window-end")
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [0.1, 0.8, 0.3]
    assert "window end" in note
    gt = np.array([0] * 10 + [1] * 10 + [0] * 10)
    timeline = cases.case_timeline(case, rows, gt, (0, 3))
    assert [r["gt_overlap"] for r in timeline] == [0, 1, 0]
    assert [r["gt_at_window_end"] for r in timeline] == [0, 1, 0]


def test_window_end_curve_holds_last_emitted_score():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event = _event(start=10, end=20, fps=10, n_frames=30)
    case = cases.SelectedCase(event, "medium", "manual", "manual")
    fig, ax = plt.subplots()
    cases._draw_curve(ax, case=case, x=np.array([1.0, 2.0]),
                      y=np.array([0.2, 0.8]), all_events=[(10, 20)],
                      context=(0, 3), score_mode="window-end")
    assert any(line.get_drawstyle() == "steps-post" for line in ax.lines)
    assert len(ax.collections) >= 1  # last emitted score persists to context end
    plt.close(fig)


def test_causal_frame_mask_and_standard_mode_warning(tmp_path):
    event = _event(start=2, end=4, fps=2, n_frames=6)
    case = cases.SelectedCase(event, "short", "manual", "manual")
    np.save(tmp_path / "video_causal_frame_scores.npy", np.array([np.nan, np.nan, 0.4, 0.4, 0.7, 0.7]))
    np.save(tmp_path / "video_causal_valid_mask.npy", np.array([0, 0, 1, 1, 1, 1], dtype=bool))
    np.save(tmp_path / "video_standard_frame_scores.npy", np.array([0.4] * 6))
    x, y, note = cases.score_points(tmp_path, case, [], (0, 3), "causal-frame")
    assert x.tolist() == [1, 1.5, 2, 2.5]
    assert y.tolist() == [0.4, 0.4, 0.7, 0.7]
    assert "source window end" in note
    _, _, standard_note = cases.score_points(tmp_path, case, [], (0, 3), "standard-frame")
    assert standard_note == "standard frame expansion; not emission-time causal alignment"


def test_manual_video_and_event_selectors():
    events = [_event("movie", i, start=i * 10, end=i * 10 + 3) for i in range(2)]
    bins = {event.case_id: "short" for event in events}
    selected, _ = cases.select_cases(events, bins, mode="manual", num_cases=12, seed=42,
                                     selectors=["movie__event01", "movie"])
    assert [case.event.event_index for case in selected] == [1, 0]
    assert all(case.selection_mode == "manual" for case in selected)


def test_video_id_file_and_manifest_video_path(tmp_path):
    selector_file = tmp_path / "selected_cases.txt"
    selector_file.write_text("# curated manually\nBad.Boys.1995__#part__event00\n\n")
    args = type("Args", (), {"video_ids": [], "video_id_file": selector_file})()
    assert cases._selectors_from_args(args) == ["Bad.Boys.1995__#part__event00"]
    video_root = tmp_path / "videos"
    video_root.mkdir()
    fallback = video_root / "some_file.mp4"
    fallback.touch()
    logical_id = "Bad.Boys.1995__#part"
    assert cases.resolve_video_path(video_root, logical_id, {"video_path": "some_file.mp4"}) == fallback
    preferred = video_root / f"{logical_id}.mp4"
    preferred.touch()
    assert cases.resolve_video_path(video_root, logical_id, {"video_path": "some_file.mp4"}) == preferred


def test_metadata_and_index_serialization_without_auc_ap(tmp_path):
    event = _event("Bad.Boys.1995__#part", start=10, end=20)
    case = cases.SelectedCase(event, "medium", "stratified-random", "GT only", 2)
    anchors = [("start", 10), ("middle", 15), ("end", 19)]
    metadata = cases.case_metadata(
        case, seed=42, score_mode="window-end", context=(0, 3), anchors=anchors,
        duplicates=[], source_window_score_csv=tmp_path / "scores.csv",
        source_gt=tmp_path / "gt.txt", source_video=tmp_path / "video.mp4",
        commit="abc123", score_note="window score emitted at window end",
    )
    assert metadata["selection_rank_within_random_permutation"] == 2
    assert metadata["selected_frame_indices"] == [10, 15, 19]
    assert metadata["git_commit"] == "abc123"
    assert metadata["git_worktree_dirty"] is False
    assert not {"auc", "ap", "selection_score"}.intersection(metadata)
    row = cases.index_row(case, seed=42, output_path=tmp_path / event.case_id)
    cases.write_csv(tmp_path / "index.csv", cases.INDEX_FIELDS, [row])
    with (tmp_path / "index.csv").open(newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["case_id"] == event.case_id
    assert "auc" not in parsed[0] and "ap" not in parsed[0]


def test_full_export_with_synthetic_decoded_anchors(tmp_path, monkeypatch):
    video_id = "Bad.Boys.1995__#part"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({video_id: {"n_frames": 30, "fps": 10.0}}))
    gt_root, infer_dir, video_root, output_dir = (
        tmp_path / "gt", tmp_path / "infer", tmp_path / "videos", tmp_path / "out",
    )
    _write_gt(gt_root, video_id, [0] * 10 + [1] * 10 + [0] * 10)
    infer_dir.mkdir()
    video_root.mkdir()
    (video_root / f"{video_id}.mp4").touch()
    _write_windows(infer_dir / f"{video_id}_window_scores.csv", video_id, [0.1, 0.8, 0.2])
    decoded = []

    def fake_decode(path, indices, n_frames):
        decoded.extend(indices)
        return [np.full((24, 32, 3), 80 + index, dtype=np.uint8) for index in indices]

    monkeypatch.setattr(cases, "decode_selected_frames", fake_decode)
    cases.main([
        "--infer-dir", str(infer_dir), "--video-root", str(video_root),
        "--gt-root", str(gt_root), "--manifest", str(manifest_path),
        "--output-dir", str(output_dir), "--selection-mode", "manual",
        "--video-ids", f"{video_id}__event00", "--dpi", "72",
    ])
    case_dir = output_dir / f"{video_id}__event00"
    assert len(decoded) == 7
    for name in ("metadata.json", "timeline.csv", "score_curve.pdf", "score_curve.png",
                 "case_panel.pdf", "case_panel.png"):
        assert (case_dir / name).is_file()
    assert len(list((case_dir / "frames").glob("*.png"))) == 7
    assert (output_dir / "index.csv").is_file()
    assert (output_dir / "selection.json").is_file()
    assert "%23" in (output_dir / "gallery.html").read_text()
    metadata = json.loads((case_dir / "metadata.json").read_text())
    assert metadata["source_video"] == str(video_root / f"{video_id}.mp4")
    assert metadata["score_mode"] == "window-end"
    assert metadata["selection_mode"] == "manual"


def test_auto_export_fails_if_any_gt_event_video_has_no_scores(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"v": {"n_frames": 5, "fps": 10}}))
    gt_root = tmp_path / "gt"
    _write_gt(gt_root, "v", [0, 1, 1, 0, 0])
    with pytest.raises(FileNotFoundError, match="selection aborted without filtering candidates"):
        cases.main([
            "--infer-dir", str(tmp_path / "missing"), "--video-root", str(tmp_path),
            "--gt-root", str(gt_root), "--manifest", str(manifest),
            "--output-dir", str(tmp_path / "out"),
        ])
