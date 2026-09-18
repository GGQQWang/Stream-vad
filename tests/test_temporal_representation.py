import json
import math

import numpy as np
import pytest

from temporal_representation import (
    CACHE_SCHEMA_VERSION,
    classify_windows,
    evaluate_representation_cache,
    fisher_statistics,
    k_block_representations,
    l2_normalize,
    load_repr_video,
    save_json,
    save_repr_video,
    variation_groups,
)


def _unit_windows(n):
    return np.arange(n), np.arange(n), np.arange(1, n + 1)


def _classify(gt):
    index, start, end = _unit_windows(len(gt))
    labels, segments = classify_windows(np.asarray(gt), index, start, end)
    return labels, segments, start, end


def test_l2_normalization_is_stable_for_nonzero_and_zero_rows():
    values = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(values[0], [0.6, 0.8])
    assert np.array_equal(values[1], [0.0, 0.0])


def test_k1_blocks_keep_pure_normal_and_anomaly_windows():
    labels, segments, _, _ = _classify([0, 0, 1, 1])
    hidden = l2_normalize(np.eye(4, dtype=np.float32))
    normal, anomaly = k_block_representations(hidden, labels, segments, 1)
    assert normal.shape == (2, 4)
    assert anomaly.shape == (2, 4)


def test_k_blocks_are_built_inside_pure_normal_segment():
    labels, segments, _, _ = _classify([0, 0, 0, 1])
    hidden = l2_normalize(np.eye(4, dtype=np.float32))
    normal, anomaly = k_block_representations(hidden, labels, segments, 2)
    assert normal.shape == (2, 4)
    assert anomaly.shape == (0, 4)


def test_k_blocks_are_built_inside_single_anomaly_event():
    labels, segments, _, _ = _classify([0, 1, 1, 1])
    hidden = l2_normalize(np.eye(4, dtype=np.float32))
    normal, anomaly = k_block_representations(hidden, labels, segments, 2)
    assert normal.shape == (0, 4)
    assert anomaly.shape == (2, 4)


def test_block_crossing_normal_anomaly_boundary_is_excluded():
    labels, segments, _, _ = _classify([0, 0, 1, 1])
    hidden = l2_normalize(np.eye(4, dtype=np.float32))
    normal, anomaly = k_block_representations(hidden, labels, segments, 3)
    assert len(normal) == 0
    assert len(anomaly) == 0


def test_block_cannot_cross_two_anomaly_events():
    labels, segments, _, _ = _classify([1, 1, 0, 1, 1])
    hidden = l2_normalize(np.eye(5, dtype=np.float32))
    normal, anomaly = k_block_representations(hidden, labels, segments, 4)
    assert len(normal) == 0
    assert len(anomaly) == 0


def test_fisher_scatter_and_formula_are_exact():
    normal = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    anomaly = np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    stats = fisher_statistics(normal, anomaly)
    assert stats["normal_scatter"] == pytest.approx(1.0)
    assert stats["anomaly_scatter"] == pytest.approx(1.0)
    assert stats["between_scatter"] == pytest.approx(8.0)
    assert stats["fdr"] == pytest.approx(4.0)


def test_fdr_increases_when_classes_are_more_separated():
    normal = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    mixed_anomaly = np.array([[0.8, 0.2], [0.7, 0.3]], dtype=np.float32)
    separated_anomaly = np.array([[-1.0, 0.0], [-0.9, -0.1]], dtype=np.float32)
    assert (
        fisher_statistics(normal, separated_anomaly)["fdr"]
        > fisher_statistics(normal, mixed_anomaly)["fdr"]
    )


def test_randomly_mixed_classes_have_lower_fdr_than_separated_classes():
    rng = np.random.default_rng(11)
    mixed_normal = l2_normalize(rng.normal(size=(32, 8)).astype(np.float32))
    mixed_anomaly = l2_normalize(rng.normal(size=(32, 8)).astype(np.float32))
    separated_normal = l2_normalize(
        np.tile([1.0] + [0.0] * 7, (32, 1)) + 0.01 * rng.normal(size=(32, 8))
    )
    separated_anomaly = l2_normalize(
        np.tile([-1.0] + [0.0] * 7, (32, 1)) + 0.01 * rng.normal(size=(32, 8))
    )
    assert (
        fisher_statistics(separated_normal, separated_anomaly)["fdr"]
        > fisher_statistics(mixed_normal, mixed_anomaly)["fdr"]
    )


def test_normal_anomaly_and_boundary_variation_toy_case():
    gt = np.array([0, 0, 1, 1])
    labels, segments, start, end = _classify(gt)
    hidden = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    groups = variation_groups(hidden, labels, segments, start, end, gt, 1)
    assert list(groups["normal"].values())[0].item() == pytest.approx(0.0)
    assert list(groups["anomaly"].values())[0].item() == pytest.approx(0.0)
    assert list(groups["boundary"].values())[0].item() == pytest.approx(2.0)


def test_pair_crossing_multiple_boundaries_is_excluded():
    gt = np.array([0, 1, 0, 1])
    labels, segments, start, end = _classify(gt)
    hidden = l2_normalize(np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32))
    groups = variation_groups(hidden, labels, segments, start, end, gt, 3)
    assert groups["boundary"] == {}


def test_k_larger_than_timeline_returns_no_pairs_or_blocks():
    gt = np.array([0, 0, 1])
    labels, segments, start, end = _classify(gt)
    hidden = l2_normalize(np.eye(3, dtype=np.float32))
    normal, anomaly = k_block_representations(hidden, labels, segments, 8)
    groups = variation_groups(hidden, labels, segments, start, end, gt, 8)
    assert normal.shape == anomaly.shape == (0, 3)
    assert groups == {"normal": {}, "anomaly": {}, "boundary": {}}


def test_normal_segment_macro_is_not_pair_micro():
    gt = np.array([0, 0, 1, 0, 0, 0])
    labels, segments, start, end = _classify(gt)
    hidden = l2_normalize(np.array([
        [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [-1, 0],
    ], dtype=np.float32))
    groups = variation_groups(hidden, labels, segments, start, end, gt, 1)["normal"]
    segment_means = [float(np.mean(values)) for values in groups.values()]
    macro = float(np.mean(segment_means))
    micro = float(np.mean(np.concatenate(list(groups.values()))))
    assert len(groups) == 2
    assert macro != pytest.approx(micro)


def test_anomaly_event_macro_preserves_event_boundaries():
    gt = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    labels, segments, start, end = _classify(gt)
    hidden = l2_normalize(np.array([
        [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [-1, 0], [0, -1],
    ], dtype=np.float32))
    groups = variation_groups(hidden, labels, segments, start, end, gt, 1)["anomaly"]
    assert len(groups) == 2
    assert sorted(len(values) for values in groups.values()) == [1, 2]


def test_boundary_variation_is_grouped_by_true_frame_boundary():
    gt = np.array([0, 0, 1, 1, 0, 0])
    labels, segments, start, end = _classify(gt)
    hidden = l2_normalize(np.array([
        [1, 0], [1, 0], [-1, 0], [-1, 0], [0, 1], [0, 1],
    ], dtype=np.float32))
    groups = variation_groups(hidden, labels, segments, start, end, gt, 2)["boundary"]
    assert {key[0] for key in groups} == {2, 4}
    assert {key[1] for key in groups} == {"0->1", "1->0"}


def test_boundary_macro_is_not_pair_micro():
    gt = np.array([0, 0, 0, 1, 1, 1, 0])
    labels, segments, start, end = _classify(gt)
    hidden = l2_normalize(np.array([
        [1, 0], [1, 0], [1, 0], [-1, 0], [0, 1], [-1, 0], [1, 0],
    ], dtype=np.float32))
    groups = variation_groups(hidden, labels, segments, start, end, gt, 2)["boundary"]
    assert sorted(len(values) for values in groups.values()) == [1, 2]
    macro = float(np.mean([np.mean(values) for values in groups.values()]))
    micro = float(np.mean(np.concatenate(list(groups.values()))))
    assert macro != pytest.approx(micro)


def test_mixed_window_and_padding_are_excluded_or_rejected():
    gt = np.array([0, 0, 1, 1])
    labels, segments = classify_windows(
        gt,
        np.array([0, 1]),
        np.array([0, 3]),
        np.array([3, 4]),
    )
    assert labels.tolist() == [-1, 1]
    assert segments.tolist()[0] == -1
    with pytest.raises(ValueError, match="inference window"):
        classify_windows(
            gt,
            np.array([0, 1, 2]),
            np.array([0, 3, -1]),
            np.array([3, 4, -1]),
        )


def test_metric_functions_are_deterministic():
    rng = np.random.default_rng(7)
    hidden = l2_normalize(rng.normal(size=(12, 5)).astype(np.float32))
    labels, segments, start, end = _classify([0] * 6 + [1] * 6)
    first = variation_groups(hidden, labels, segments, start, end, np.r_[np.zeros(6), np.ones(6)], 2)
    second = variation_groups(hidden, labels, segments, start, end, np.r_[np.zeros(6), np.ones(6)], 2)
    for category in first:
        assert first[category].keys() == second[category].keys()
        for key in first[category]:
            assert np.array_equal(first[category][key], second[category][key])


@pytest.mark.parametrize("method", ("ssm", "lstm", "stc", "qformer", "transformer"))
def test_all_model_caches_use_the_same_schema(tmp_path, method):
    path = tmp_path / f"{method}.npz"
    save_repr_video(
        path,
        method=method,
        video_id="v",
        n_frames=2,
        fps=30.0,
        window_index=np.array([0, 1]),
        start_frame=np.array([0, 1]),
        valid_end_frame=np.array([1, 2]),
        score_hidden=np.eye(2, dtype=np.float32),
        storage_dtype="float16",
    )
    loaded = load_repr_video(path)
    assert loaded["storage_dtype"] == "float16"
    assert loaded["method"] == method
    assert loaded["score_hidden"].dtype == np.float32
    assert loaded["window_index"].tolist() == [0, 1]


def _write_eval_cache(root, gt_root, method, videos):
    root.mkdir()
    entries = []
    for video_id, gt, hidden in videos:
        np.savetxt(gt_root / f"{video_id}.txt", gt, fmt="%d")
        index, start, end = _unit_windows(len(gt))
        path = root / f"{video_id}.npz"
        save_repr_video(
            path,
            method=method,
            video_id=video_id,
            n_frames=len(gt),
            fps=1.0,
            window_index=index,
            start_frame=start,
            valid_end_frame=end,
            score_hidden=np.asarray(hidden, dtype=np.float32),
            storage_dtype="float32",
        )
        entries.append({"video_id": video_id, "file": path.name})
    save_json(root / "manifest.json", {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "method": method,
        "videos": entries,
    })


def test_fdr_uses_video_macro_not_global_sample_micro(tmp_path):
    gt_root = tmp_path / "gt"
    gt_root.mkdir()
    cache = tmp_path / "cache"
    videos = [
        ("a", [0, 0, 1, 1], [[1, 0], [0.9, 0.1], [-1, 0], [-0.9, -0.1]]),
        ("b", [0, 0, 1, 1], [[1, 0], [0, 1], [0.2, 0.8], [0, 1]]),
    ]
    _write_eval_cache(cache, gt_root, "lstm", videos)
    row = evaluate_representation_cache(cache, gt_root, [1])[0]
    per_video = []
    pooled_normal = []
    pooled_anomaly = []
    for _, gt, hidden in videos:
        labels, segments, _, _ = _classify(gt)
        normal, anomaly = k_block_representations(l2_normalize(hidden), labels, segments, 1)
        per_video.append(fisher_statistics(normal, anomaly)["fdr"])
        pooled_normal.append(normal)
        pooled_anomaly.append(anomaly)
    pooled = fisher_statistics(np.concatenate(pooled_normal), np.concatenate(pooled_anomaly))["fdr"]
    assert row["fdr"] == pytest.approx(np.mean(per_video))
    assert row["fdr"] != pytest.approx(pooled)


def test_k64_insufficient_coverage_returns_nan_and_zero_counts(tmp_path):
    gt_root = tmp_path / "gt"
    gt_root.mkdir()
    cache = tmp_path / "cache"
    _write_eval_cache(
        cache,
        gt_root,
        "ssm",
        [("v", [0, 0, 1, 1], [[1, 0], [1, 0], [-1, 0], [-1, 0]])],
    )
    row = evaluate_representation_cache(cache, gt_root, [64])[0]
    assert math.isnan(row["fdr"])
    assert math.isnan(row["v_anomaly"])
    assert row["fdr_valid_videos"] == 0
    assert row["anomaly_block_count"] == 0
    assert row["anomaly_pair_count"] == 0
    assert row["fdr_skipped_videos"] == [{
        "video_id": "v",
        "reason": "insufficient_both_classes",
        "normal_blocks": 0,
        "anomaly_blocks": 0,
    }]
