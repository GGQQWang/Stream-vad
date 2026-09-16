import numpy as np
import pytest

from stage_ap import (
    aggregate_event_metrics,
    deterministic_uniform_sample,
    evaluate_video_stage_ap,
    ranking_average_precision,
    split_event_windows,
)


def _rows(n_frames):
    return [
        {"window_index": index, "start_frame": index, "end_frame": index + 1}
        for index in range(n_frames)
    ]


def _hidden(n_frames, dim=2):
    values = np.arange(n_frames * dim, dtype=np.float32).reshape(n_frames, dim) + 1.0
    return values


def test_split_event_windows_equal_thirds():
    early, middle, late = split_event_windows([3, 4, 5, 6, 7, 8])
    assert early.tolist() == [3, 4]
    assert middle.tolist() == [5, 6]
    assert late.tolist() == [7, 8]


def test_split_event_windows_remainder_goes_to_earlier_thirds():
    early, middle, late = split_event_windows([0, 1, 2, 3, 4, 5, 6, 7])
    assert early.tolist() == [0, 1, 2]
    assert middle.tolist() == [3, 4, 5]
    assert late.tolist() == [6, 7]


def test_adjacent_pre_post_normal_windows_are_selected():
    gt = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
    record = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=_hidden(len(gt)),
    )[0]
    assert record["status"] == "valid"
    assert record["window_indices"]["pre"] == [1]
    assert record["window_indices"]["post"] == [5]


def test_adjacent_normal_windows_do_not_cross_another_event():
    gt = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.int64)
    records = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=_hidden(len(gt)),
    )
    assert len(records) == 2
    assert records[0]["window_indices"]["post"] == [4]
    assert records[1]["window_indices"]["pre"] == [5]
    assert records[0]["window_indices"]["post"][-1] < records[1]["start"]
    assert records[1]["window_indices"]["pre"][0] >= records[0]["end"]


def test_event_at_video_start_without_pre_is_skipped():
    gt = np.array([1, 1, 1, 0], dtype=np.int64)
    record = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=_hidden(len(gt)),
    )[0]
    assert record["status"] == "skipped"
    assert record["skip_reason"] == "missing_adjacent_pre_normal"


def test_event_at_video_end_without_post_is_skipped():
    gt = np.array([0, 1, 1, 1], dtype=np.int64)
    record = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=_hidden(len(gt)),
    )[0]
    assert record["status"] == "skipped"
    assert record["skip_reason"] == "missing_adjacent_post_normal"


def test_m_is_minimum_of_all_five_regions():
    gt = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int64)
    record = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=_hidden(len(gt)),
    )[0]
    assert record["status"] == "valid"
    assert record["m"] == 1
    assert all(len(indices) == 1 for indices in record["window_indices"].values())


def test_deterministic_uniform_sampling_spans_input():
    first = deterministic_uniform_sample([10, 11, 12, 13, 14, 15, 16], 3)
    second = deterministic_uniform_sample([10, 11, 12, 13, 14, 15, 16], 3)
    assert first.tolist() == [10, 13, 16]
    assert np.array_equal(first, second)


def test_ranking_ap_toy_separable_and_tied_cases():
    query = np.array([1.0, 0.0])
    positives = np.array([[1.0, 0.0], [0.9, 0.1]])
    negatives = np.array([[-1.0, 0.0], [-0.9, -0.1], [0.0, 1.0], [0.0, -1.0]])
    assert ranking_average_precision(query, positives, negatives) == pytest.approx(1.0)

    tied = np.ones((6, 2), dtype=np.float32)
    tied_ap = ranking_average_precision(tied[0], tied[:2], tied[2:])
    assert tied_ap == pytest.approx(1.0 / 3.0)
    assert tied_ap < 0.9


def test_full_toy_stage_ap_is_high_only_when_stages_are_distinct_from_normal():
    gt = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
    separable = np.array([
        [-1.0, 0.0], [-1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [-1.0, 0.0], [-1.0, 0.0],
    ])
    high = evaluate_video_stage_ap(
        video_id="a", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=separable,
    )[0]
    assert high["stage_ap"] == pytest.approx(1.0)

    tied = evaluate_video_stage_ap(
        video_id="b", frame_gt=gt, window_rows=_rows(len(gt)), score_hidden=np.ones_like(separable),
    )[0]
    assert tied["stage_ap"] == pytest.approx(1.0 / 3.0)


def test_dataset_aggregation_is_event_macro_not_query_micro():
    records = [
        {"status": "valid", "ap_em": 1.0, "ap_ml": 1.0, "ap_el": 1.0, "stage_ap": 1.0},
        {"status": "valid", "ap_em": 0.0, "ap_ml": 0.0, "ap_el": 0.0, "stage_ap": 0.0},
    ]
    summary = aggregate_event_metrics(records)
    query_micro_for_unequal_event_lengths = (1.0 + 0.0 + 0.0 + 0.0) / 4.0
    assert summary["stage_ap"] == pytest.approx(0.5)
    assert summary["stage_ap"] != pytest.approx(query_micro_for_unequal_event_lengths)


def test_window_mapping_uses_exact_half_open_inference_spans():
    gt = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.int64)
    rows = [
        {"window_index": 0, "start_frame": 0, "end_frame": 4},
        {"window_index": 1, "start_frame": 4, "end_frame": 8},
        {"window_index": 2, "start_frame": 8, "end_frame": 10},
    ]
    record = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=rows, score_hidden=_hidden(3),
    )[0]
    assert record["mapped_window_indices"] == [1]
    assert record["skip_reason"] == "fewer_than_three_event_windows"


def test_window_overlapping_two_events_is_excluded_from_both():
    gt = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int64)
    rows = [
        {"window_index": 0, "start_frame": 0, "end_frame": 2},
        {"window_index": 1, "start_frame": 2, "end_frame": 6},
        {"window_index": 2, "start_frame": 6, "end_frame": 8},
    ]
    records = evaluate_video_stage_ap(
        video_id="v", frame_gt=gt, window_rows=rows, score_hidden=_hidden(3),
    )
    assert records[0]["mapped_window_indices"] == [1]
    assert records[0]["ambiguous_windows_excluded"] == 1
    assert records[0]["skip_reason"] == "no_unambiguous_event_windows"
    assert records[1]["mapped_window_indices"] == [1, 2]
    assert records[1]["ambiguous_windows_excluded"] == 1
