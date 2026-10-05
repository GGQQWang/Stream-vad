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
    timeline = cases.case_timeline(case, rows, gt)
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
                      plot_context=(0, 3), score_mode="window-end")
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
        case, seed=42, score_mode="window-end", plot_range="full",
        frame_context=(0, 3), plot_context=(0, 5),
        num_gt_events_in_video=1, local_zoom_exported=False, anchors=anchors,
        duplicates=[], source_window_score_csv=tmp_path / "scores.csv",
        source_gt=tmp_path / "gt.txt", source_video=tmp_path / "video.mp4",
        commit="abc123", score_note="window score emitted at window end",
    )
    assert metadata["selection_rank_within_random_permutation"] == 2
    assert metadata["selected_frame_indices"] == [10, 15, 19]
    assert metadata["git_commit"] == "abc123"
    assert metadata["git_worktree_dirty"] is False
    assert metadata["plot_context_end_sec"] == 5
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


def _auc_fixture(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    gt_root, infer_dir = tmp_path / "gt", tmp_path / "infer"
    infer_dir.mkdir()
    aucs = [0.99, 0.95, 0.80, 0.30, 0.20, 0.10]
    ids = [f"v{i}" for i in range(len(aucs))] + ["normal", "undefined"]
    manifest = {video_id: {"n_frames": 60, "fps": 10.0} for video_id in ids}
    manifest_path.write_text(json.dumps(manifest))
    videos = []
    for i, video_id in enumerate(ids):
        gt = [0] * 60
        if video_id != "normal":
            gt[10:11 + i] = [1] * (i + 1)
            if video_id == "v0":
                gt[30:34] = [1] * 4
        _write_gt(gt_root, video_id, gt)
        _write_windows(infer_dir / f"{video_id}_window_scores.csv", video_id, [0.5] * 6)
        videos.append({
            "video_id": video_id,
            "standard_auc": aucs[i] if i < len(aucs) else 0.99 if video_id == "normal" else None,
            "standard_ap": 0.5, "causal_auc": 0.4, "causal_ap": 0.3,
        })
    (infer_dir / "metrics.json").write_text(json.dumps({"videos": videos}))
    args = [
        "--infer-dir", str(infer_dir), "--video-root", str(tmp_path),
        "--gt-root", str(gt_root), "--manifest", str(manifest_path),
        "--output-dir", str(tmp_path / "out"), "--selection-mode", "high-auc-pool",
    ]
    return manifest_path, gt_root, infer_dir, args


def test_high_auc_metrics_reading_and_none_or_normal_exclusion(tmp_path):
    manifest_path, gt_root, infer_dir, _ = _auc_fixture(tmp_path)
    manifest = cases.load_manifest(manifest_path)
    events = cases.build_candidates(manifest, gt_root)
    records = cases.load_video_metrics(infer_dir / "metrics.json", events, manifest)
    assert [record["video_id"] for record in records] == [f"v{i}" for i in range(6)]
    assert [record["auc_rank"] for record in records] == list(range(1, 7))
    assert records[0]["num_gt_events"] == 2
    assert records[0]["standard_ap"] == 0.5
    (infer_dir / "bad.json").write_text(json.dumps({"videos": {}}))
    with pytest.raises(ValueError, match="expected metrics"):
        cases.load_video_metrics(infer_dir / "bad.json", events, manifest)


def test_high_auc_top_k_min_and_multi_event_pool(tmp_path):
    manifest_path, gt_root, infer_dir, _ = _auc_fixture(tmp_path)
    manifest = cases.load_manifest(manifest_path)
    events = cases.build_candidates(manifest, gt_root)
    records = cases.load_video_metrics(infer_dir / "metrics.json", events, manifest)
    selected, audit, rows = cases.select_high_auc_cases(
        events, records, infer_dir=infer_dir, num_cases=0, seed=42,
        auc_top_k=2, auc_min=0.9, num_failures=0, failure_bottom_k=30, failure_seed=42,
    )
    assert [record["video_id"] for record in audit["success_candidate_videos"]] == ["v0", "v1"]
    assert audit["success_event_pool_size"] == 3
    assert {case.event.case_id for case in selected} == {"v0__event00", "v0__event01", "v1__event00"}
    assert [row["pool_role"] for row in rows] == ["success_candidate"] * 2 + ["unused"] * 4
    _, filtered, _ = cases.select_high_auc_cases(
        events, records, infer_dir=infer_dir, num_cases=0, seed=42,
        auc_top_k=None, auc_min=0.95, num_failures=0, failure_bottom_k=30, failure_seed=42,
    )
    assert filtered["success_video_pool_size"] == 2


def test_high_auc_duration_stratification_is_seeded(tmp_path):
    manifest_path, gt_root, infer_dir, _ = _auc_fixture(tmp_path)
    manifest = cases.load_manifest(manifest_path)
    events = cases.build_candidates(manifest, gt_root)
    records = cases.load_video_metrics(infer_dir / "metrics.json", events, manifest)
    def select(order):
        return cases.select_high_auc_cases(
            order, records, infer_dir=infer_dir, num_cases=3, seed=17,
            auc_top_k=4, auc_min=None, num_failures=0, failure_bottom_k=30, failure_seed=17,
        )[0]
    first, second = select(events), select(list(reversed(events)))
    assert [case.event.case_id for case in first] == [case.event.case_id for case in second]
    assert {case.duration_bin for case in first} == set(cases.BINS)
    assert all(case.selection_mode == "high-auc-pool" and case.case_role == "success" for case in first)


def test_failure_bottom_k_seeded_and_median_event(tmp_path):
    manifest_path, gt_root, infer_dir, _ = _auc_fixture(tmp_path)
    manifest = cases.load_manifest(manifest_path)
    events = cases.build_candidates(manifest, gt_root)
    records = cases.load_video_metrics(infer_dir / "metrics.json", events, manifest)
    def select():
        return cases.select_high_auc_cases(
            events, records, infer_dir=infer_dir, num_cases=1, seed=42,
            auc_top_k=2, auc_min=None, num_failures=2, failure_bottom_k=2, failure_seed=7,
        )
    selected, audit, rows = select()
    assert [record["video_id"] for record in audit["failure_pool_videos"]] == ["v5", "v4"]
    assert [case.event.video_id for case in selected if case.case_role == "failure"] == [
        case.event.video_id for case in select()[0] if case.case_role == "failure"
    ]
    assert {case.event.video_id for case in selected if case.case_role == "failure"} == {"v4", "v5"}
    assert [row["pool_role"] for row in rows[-2:]] == ["failure_candidate"] * 2
    assert all(case.sampling_seed == 7 for case in selected if case.case_role == "failure")


def test_failure_uses_event_closest_to_video_median_duration(tmp_path):
    events = [_event("high", end=12)] + [
        _event("low", index, start=10 + index * 15, end=10 + index * 15 + duration,
               n_frames=70)
        for index, duration in enumerate((2, 5, 11))
    ]
    records = [
        {"video_id": "high", "standard_auc": 0.9, "auc_rank": 1, "num_gt_events": 1},
        {"video_id": "low", "standard_auc": 0.1, "auc_rank": 2, "num_gt_events": 3},
    ]
    for video_id in ("high", "low"):
        (tmp_path / f"{video_id}_window_scores.csv").touch()
    selected, _, _ = cases.select_high_auc_cases(
        events, records, infer_dir=tmp_path, num_cases=1, seed=1,
        auc_top_k=1, auc_min=None, num_failures=1, failure_bottom_k=1, failure_seed=1,
    )
    failure = next(case for case in selected if case.case_role == "failure")
    assert failure.event.event_index == 1


def test_high_auc_pool_missing_window_scores_fails_even_in_dry_run(tmp_path):
    _, _, infer_dir, args = _auc_fixture(tmp_path)
    (infer_dir / "v0_window_scores.csv").unlink()
    with pytest.raises(FileNotFoundError, match="v0"):
        cases.main(args + ["--auc-top-k", "1", "--dry-run"])
    (infer_dir / "v0_window_scores.csv").touch()
    (infer_dir / "v5_window_scores.csv").unlink()
    with pytest.raises(FileNotFoundError, match="v5"):
        cases.main(args + ["--auc-top-k", "1", "--num-failures", "1",
                           "--failure-bottom-k", "1", "--dry-run"])


def test_high_auc_dry_run_audit_does_not_write_outputs(tmp_path, capsys):
    _, _, _, args = _auc_fixture(tmp_path)
    cases.main(args + ["--auc-top-k", "2", "--num-cases", "2", "--dry-run"])
    audit = json.loads(capsys.readouterr().out)
    assert audit["selection_mode"] == "high-auc-pool"
    assert audit["selection_basis"] == cases.AUC_SELECTION_BASIS
    assert audit["success_video_pool_size"] == 2
    assert audit["success_event_pool_size"] == 3
    assert audit["num_selected_success"] == 2
    assert len(audit["success_candidate_videos"]) == 2
    assert not (tmp_path / "out").exists()


def test_high_auc_export_audit_metadata_index_candidates_and_gallery(tmp_path, monkeypatch):
    _, _, _, args = _auc_fixture(tmp_path)
    def fake_export(case, **kwargs):
        case_dir = kwargs["output_dir"] / case.event.case_id
        case_dir.mkdir()
        (case_dir / "metadata.json").write_text(json.dumps(cases.case_metadata(
            case, seed=kwargs["seed"], score_mode="window-end", plot_range="full",
            frame_context=(0, 6), plot_context=(0, 6),
            num_gt_events_in_video=1, local_zoom_exported=False,
            anchors=[], duplicates=[], source_window_score_csv=tmp_path / "scores.csv",
            source_gt=tmp_path / "gt.txt", source_video=tmp_path / "video.mp4",
            commit="test", score_note="window-end",
        )))
        return cases.index_row(case, seed=kwargs["seed"], output_path=case_dir)
    monkeypatch.setattr(cases, "export_case", fake_export)
    cases.main(args + ["--auc-top-k", "2", "--num-cases", "2", "--num-failures", "1",
                       "--failure-bottom-k", "2", "--failure-seed", "7"])
    output = tmp_path / "out"
    selection = json.loads((output / "selection.json").read_text())
    assert selection["selection_basis"] == cases.AUC_SELECTION_BASIS
    assert selection["metrics_path"].endswith("metrics.json")
    assert selection["num_selected_success"] == 2
    assert selection["num_failures"] == 1
    with (output / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_role"] for row in rows] == ["success", "success", "failure"]
    assert float(rows[0]["video_standard_auc"]) >= float(rows[1]["video_standard_auc"])
    assert all(row["selection_basis"] == cases.AUC_SELECTION_BASIS for row in rows)
    success_meta = json.loads((output / rows[0]["case_id"] / "metadata.json").read_text())
    failure_meta = json.loads((output / rows[-1]["case_id"] / "metadata.json").read_text())
    assert success_meta["success_pool_rank"] == success_meta["auc_rank"]
    assert success_meta["success_pool_size"] == 2
    assert success_meta["success_selection_reason"]
    assert failure_meta["failure_pool_bottom_k"] == 2
    assert failure_meta["failure_selection_reason"]
    assert failure_meta["seed"] == 7
    with (output / "candidate_videos.csv").open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert len(candidates) == 6
    assert candidates[0]["pool_role"] == "success_candidate"
    assert candidates[-1]["pool_role"] == "failure_candidate"
    assert candidates[0]["num_gt_events"] == "2"
    gallery = (output / "gallery.html").read_text()
    assert "success" in gallery and "failure" in gallery and "standard AUC" in gallery
    assert gallery.index("success") < gallery.index("failure")


def test_high_auc_overlap_rejects_unavailable_distinct_failures(tmp_path):
    manifest_path, gt_root, infer_dir, _ = _auc_fixture(tmp_path)
    manifest = cases.load_manifest(manifest_path)
    events = cases.build_candidates(manifest, gt_root)
    records = cases.load_video_metrics(infer_dir / "metrics.json", events, manifest)
    with pytest.raises(ValueError, match="outside the success pool"):
        cases.select_high_auc_cases(
            events, records, infer_dir=infer_dir, num_cases=1, seed=1,
            auc_top_k=6, auc_min=None, num_failures=1, failure_bottom_k=2, failure_seed=1,
        )


def test_full_and_local_plot_context_do_not_change_event_frame_anchors():
    event = _event(start=30, end=40, fps=10, n_frames=100)
    frame_context = cases.context_interval(event, pre_sec=1, post_sec=2)
    anchors, _ = cases.temporal_anchors(event, pre_sec=1, post_sec=2, frames_per_case=7)
    assert frame_context == (2, 6)
    assert cases.plot_context_interval(event, frame_context, "full") == (0, 10)
    assert cases.plot_context_interval(event, frame_context, "local") == frame_context
    assert anchors[0] == ("pre", 25)
    assert anchors[-1] == ("post", 49)
    assert [label for label, _ in anchors[1:-1]] == ["start", "early", "middle", "late", "end"]


def test_full_window_end_scores_and_timeline_include_all_windows(tmp_path):
    event = _event(start=30, end=40, fps=10, n_frames=100)
    case = cases.SelectedCase(event, "medium", "manual", "manual")
    path = tmp_path / "video_window_scores.csv"
    scores = [i / 10 for i in range(10)]
    _write_windows(path, "video", scores)
    rows = cases.read_window_scores(path, video_id="video", n_frames=100, fps=10)
    full = cases.plot_context_interval(event, (2, 6), "full")
    full_x, full_y, _ = cases.score_points(tmp_path, case, rows, full, "window-end")
    local_x, local_y, _ = cases.score_points(tmp_path, case, rows, (2, 6), "window-end")
    assert full_x.tolist() == list(range(1, 11))
    assert full_y.tolist() == scores
    assert local_x.tolist() == [2, 3, 4, 5, 6]
    assert local_y.tolist() == scores[1:6]
    gt = np.array([0] * 30 + [1] * 10 + [0] * 60)
    timeline = cases.case_timeline(case, rows, gt)
    assert len(timeline) == 10
    assert [row["window_index"] for row in timeline] == list(range(10))
    assert [row["gt_overlap"] for row in timeline] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_full_frame_score_modes_include_entire_valid_video(tmp_path):
    event = _event(start=10, end=20, fps=10, n_frames=30)
    case = cases.SelectedCase(event, "medium", "manual", "manual")
    np.save(tmp_path / "video_causal_frame_scores.npy", np.arange(30) / 30)
    np.save(tmp_path / "video_causal_valid_mask.npy", np.array([False] * 10 + [True] * 20))
    np.save(tmp_path / "video_standard_frame_scores.npy", np.arange(30) / 30)
    causal_x, _, _ = cases.score_points(tmp_path, case, [], (0, 3), "causal-frame")
    standard_x, _, _ = cases.score_points(tmp_path, case, [], (0, 3), "standard-frame")
    assert causal_x.tolist() == pytest.approx([i / 10 for i in range(10, 30)])
    assert standard_x.tolist() == pytest.approx([i / 10 for i in range(30)])


def test_full_curve_distinguishes_selected_and_other_gt_events():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event = _event(start=30, end=40, fps=10, n_frames=100)
    case = cases.SelectedCase(event, "medium", "manual", "manual")
    fig, ax = plt.subplots()
    cases._draw_curve(ax, case=case, x=np.arange(1, 11), y=np.full(10, 0.5),
                      all_events=[(10, 20), (30, 40), (70, 80)],
                      plot_context=(0, 10), score_mode="window-end")
    labels = ax.get_legend_handles_labels()[1]
    assert set(labels) == {"Selected GT event", "Other GT anomaly interval", "Anomaly score"}
    assert [patch.get_alpha() for patch in ax.patches] == [0.10, 0.20, 0.10]
    assert ax.get_xlim() == (0, 10)
    assert ax.get_ylim() == (0, 1)
    assert sum(line.get_linestyle() == "--" for line in ax.lines) == 2
    plt.close(fig)


@pytest.mark.parametrize("plot_args,expected_range,zoom", [
    ([], "full", False),
    (["--plot-range", "local"], "local", False),
    (["--export-local-zoom"], "full", True),
])
def test_export_plot_range_metadata_timeline_and_optional_zoom(
    tmp_path, monkeypatch, plot_args, expected_range, zoom,
):
    video_id = "multi.event"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({video_id: {"n_frames": 60, "fps": 10}}))
    gt_root, infer_dir, video_root, output_dir = (
        tmp_path / "gt", tmp_path / "infer", tmp_path / "videos", tmp_path / "out",
    )
    _write_gt(gt_root, video_id, [0] * 10 + [1] * 10 + [0] * 20 + [1] * 10 + [0] * 10)
    infer_dir.mkdir()
    video_root.mkdir()
    (video_root / f"{video_id}.mp4").touch()
    _write_windows(infer_dir / f"{video_id}_window_scores.csv", video_id, [0.1, 0.8, 0.2, 0.1, 0.9, 0.2])
    monkeypatch.setattr(cases, "decode_selected_frames", lambda path, indices, n_frames: [
        np.full((16, 20, 3), 80, dtype=np.uint8) for _ in indices
    ])
    cases.main([
        "--infer-dir", str(infer_dir), "--video-root", str(video_root),
        "--gt-root", str(gt_root), "--manifest", str(manifest_path),
        "--output-dir", str(output_dir), "--selection-mode", "manual",
        "--video-ids", f"{video_id}__event00", "--pre-sec", "0.5",
        "--post-sec", "0.5", "--dpi", "72", *plot_args,
    ])
    case_dir = output_dir / f"{video_id}__event00"
    metadata = json.loads((case_dir / "metadata.json").read_text())
    assert metadata["plot_range"] == expected_range
    assert (metadata["frame_context_start_sec"], metadata["frame_context_end_sec"]) == (0.5, 2.5)
    expected_plot = (0, 6) if expected_range == "full" else (0.5, 2.5)
    assert (metadata["plot_context_start_sec"], metadata["plot_context_end_sec"]) == expected_plot
    assert metadata["num_gt_events_in_video"] == 2
    assert metadata["selected_event_index"] == 0
    assert metadata["local_zoom_exported"] is zoom
    assert metadata["selected_frame_indices"][0] == 8
    assert (case_dir / "case_panel.png").is_file()
    assert (case_dir / "score_curve.pdf").is_file()
    assert (case_dir / "score_curve_local.pdf").is_file() is zoom
    assert (case_dir / "score_curve_local.png").is_file() is zoom
    with (case_dir / "timeline.csv").open(newline="") as handle:
        timeline = list(csv.DictReader(handle))
    assert len(timeline) == 6
    assert [int(row["window_index"]) for row in timeline] == list(range(6))
