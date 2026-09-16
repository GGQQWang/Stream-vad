import csv
import json

import pytest

from tools.summarize_temporal_ablation import (
    FIELDS,
    load_temporal_result,
    validate_fair_protocol,
    write_temporal_results,
)


def _write_inputs(root, method="lstm", saved_method=None):
    infer_dir = root / f"infer_ucf_temporal_{method}_epoch0"
    stage_dir = root / f"stage_ap_ucf_temporal_{method}_epoch0"
    infer_dir.mkdir()
    stage_dir.mkdir()
    (infer_dir / "metrics.json").write_text(json.dumps({
        "temporal_model": saved_method or method,
        "global_standard_auc": 0.8,
        "mean_latency_ms": 10.0,
        "p95_latency_ms": 12.0,
        "throughput_windows_s": 100.0,
        "history_elements": 512,
        "history_buffer_bytes": 2048,
        "history_buffer_mb": 2048 / (1024 ** 2),
        "temporal_params": 123,
        "trainable_params": 456,
        "peak_gpu_memory_gb": 7.5,
        "profiling_batch_size": 1,
        "profile_measured_windows": 100,
        "num_failed_videos": 0,
        "temporal_history": 16,
        "visual_fusion": "state_spatial_film",
        "inference_dtype": "torch.bfloat16",
        "test_manifest": "/data/test.json",
        "feature_cache_root": "/data/cache",
        "profile_warmup_windows": 16,
        "profile_seen_windows": 116,
        "profiling_scope": "cached_visual_feature_to_score_logit",
    }))
    (stage_dir / "stage_ap_summary.json").write_text(json.dumps({
        "em_ap": 0.1,
        "ml_ap": 0.2,
        "el_ap": 0.3,
        "stage_ap": 0.2,
        "feature": "score_hidden",
        "aggregation": "event_macro",
        "num_failed_videos": 0,
    }))


def test_temporal_summary_combines_existing_metric_outputs(tmp_path):
    _write_inputs(tmp_path)
    row = load_temporal_result(tmp_path, "lstm")
    assert row == {
        "method": "lstm",
        "auc": 0.8,
        "em_ap": 0.1,
        "ml_ap": 0.2,
        "el_ap": 0.3,
        "stage_ap": 0.2,
        "mean_latency_ms": 10.0,
        "p95_latency_ms": 12.0,
        "throughput_windows_s": 100.0,
        "history_elements": 512,
        "history_buffer_bytes": 2048,
        "history_buffer_mb": 2048 / (1024 ** 2),
        "temporal_params": 123,
        "trainable_params": 456,
        "peak_gpu_memory_gb": 7.5,
    }
    csv_path, json_path = write_temporal_results([row], tmp_path / "results")
    with open(csv_path, newline="") as f:
        assert tuple(next(csv.reader(f))) == FIELDS
    assert json.loads(json_path.read_text()) == [row]


def test_temporal_summary_rejects_method_mismatch(tmp_path):
    _write_inputs(tmp_path, saved_method="transformer")
    with pytest.raises(ValueError, match="temporal method mismatch"):
        load_temporal_result(tmp_path, "lstm")


def test_temporal_summary_validates_identical_profile_protocol(tmp_path):
    _write_inputs(tmp_path, method="lstm")
    _write_inputs(tmp_path, method="stc")
    validate_fair_protocol(tmp_path, ["lstm", "stc"])
    metrics_path = tmp_path / "infer_ucf_temporal_stc_epoch0" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["temporal_history"] = 32
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="profiling protocol mismatch"):
        validate_fair_protocol(tmp_path, ["lstm", "stc"])
