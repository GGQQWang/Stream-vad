"""CPU-only checks for the five-group ablation orchestration."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from frame_score_eval import frame_scores_from_windows, metadata_without_test_annotations
from tools.ablation_5_core import GROUPS, ZERO_SHOT_PROMPT, group_config, parse_generated_score
from tools.run_ablation_5 import (
    eval_command, latest_checkpoint, load_config, next_eval_dir, selected_groups, train_command,
)
from tools.summarize_ablation_5 import COMPARISONS, make_rows, write_results


ROOT = Path(__file__).resolve().parents[1]


def _config():
    details = {
        "train_json": "/unverified/train.json", "train_video_root": "/unverified/train_videos",
        "feature_cache_root": "/unverified/cache", "test_manifest": "/unverified/test.json",
        "test_video_root": "/unverified/test_videos", "test_gt_root": "/unverified/gt",
        "smoke_video_id": "sample",
    }
    return {"model_path": "/unverified/model", "seed": 42, "epochs": 2,
            "lr": 1e-5, "batch_size": 1, "grad_accum": 4,
            "lambda_sem": 0.1, "pixels": 100352, "ucf": dict(details), "xd": dict(details)}


def test_groups_only_a_to_e_and_exact_objectives():
    assert tuple(GROUPS) == ("A", "B", "C", "D", "E")
    for group in GROUPS:
        config = group_config(group, 0.1)
        assert config["lambda_world"] == 0
        assert config["visual_fusion"] == "state_only"
        assert config["lambda_sum"] == (0.1 if group in {"C", "E"} else 0)
        assert config["temporal_mode"] == ("none" if group in {"A", "B", "C"} else "ssm")
    for group in "FG":
        with pytest.raises(ValueError, match="unsupported"):
            group_config(group, 0.1)


@pytest.mark.parametrize("raw,score,error", [
    (" 0.75 \n", 0.75, None), ("1.0", 1.0, None),
    ("0.5 because anomaly", None, "not_a_single_number"),
    ("nan", None, "not_a_single_number"),
    ("1.2", None, "out_of_range"), ("-0.1", None, "out_of_range"),
])
def test_zero_shot_score_parser(raw, score, error):
    assert parse_generated_score(raw) == (score, error)
    assert ZERO_SHOT_PROMPT.endswith("Output only a single number without any explanation.")


def test_shared_frame_mapping_retains_existing_standard_and_causal_semantics():
    rows = [
        {"start_frame": 0, "end_frame": 3, "score_prob": 0.2},
        {"start_frame": 3, "end_frame": 6, "score_prob": 0.8},
        {"start_frame": 6, "end_frame": 8, "score_prob": 0.1},
    ]
    standard, causal, valid = frame_scores_from_windows(rows, 8)
    assert standard.tolist() == pytest.approx([0.2] * 3 + [0.8] * 3 + [0.1] * 2)
    assert valid.tolist() == [False] * 3 + [True] * 5
    assert np.isnan(causal[:3]).all()
    assert causal[3:].tolist() == pytest.approx([0.2] * 3 + [0.8] * 2)


def test_official_test_index_does_not_use_hivau_template_labels():
    meta = metadata_without_test_annotations(48, 24.0)
    assert meta["n_frames"] == 48 and meta["fps"] == 24.0
    assert meta["video_label"] == 0
    assert meta["events"] == [] and meta["events_summary_split"] == []


def test_config_validation_and_commands(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()))
    config = load_config(config_path, groups=["B", "C", "D", "E"], train=True,
                         evaluate=True, dry_run=True)
    for group in ("B", "C", "D", "E"):
        command = train_command(config, config["ucf"], group, tmp_path / "train", "cuda:0", None, 0)
        assert command[command.index("--ablation-group") + 1] == group
        assert float(command[command.index("--lambda-sum") + 1]) == (0.1 if group in "CE" else 0)
        assert command[command.index("--lambda-world") + 1] == "0"
        assert "--init-checkpoint" not in command
        inference = eval_command(config, config["ucf"], group, tmp_path / "ckpt",
                                 tmp_path / "eval", "cuda:0", "")
        assert inference[inference.index("--inference-max-windows") + 1] == "1"
        assert "--strict" in inference
        assert "--official-gt-only" in inference
    zero_shot = eval_command(config, config["ucf"], "A", None, tmp_path / "eval", "cuda:0", "")
    assert zero_shot[2] == "tools/infer_zero_shot_ablation.py"
    assert "--stage1-dir" not in zero_shot
    smoke = train_command(config, config["ucf"], "C", tmp_path / "smoke", "cuda:0", None, 2)
    assert smoke[smoke.index("--epochs") + 1] == "1"
    assert smoke[smoke.index("--max-train-updates") + 1] == "2"
    broken = _config()
    broken["xd"]["test_gt_root"] = ""
    config_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="xd missing"):
        load_config(config_path, groups=["A"], train=False, evaluate=True, dry_run=True)


def test_shell_dry_run_prints_all_five_without_side_effects(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()))
    output = tmp_path / "outputs"
    result = subprocess.run([
        "bash", str(ROOT / "scripts/run_ablation_5.sh"), "--config", str(config_path),
        "--dry-run", "--output-root", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    assert result.stdout.count("--ablation-group") == 8  # B-E, independently for UCF and XD
    assert result.stdout.count("infer_zero_shot_ablation.py") == 2
    assert "F-PESL" not in result.stdout
    assert not output.exists()


def test_ucf_only_dry_run_needs_no_xd_config_or_commands(tmp_path):
    config = _config()
    config.pop("xd")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "outputs"
    result = subprocess.run([
        "bash", str(ROOT / "scripts/run_ablation_5.sh"), "--config", str(config_path),
        "--dataset", "ucf", "--dry-run", "--output-root", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=True)
    assert "datasets=ucf" in result.stdout
    assert result.stdout.count("--ablation-group") == 4
    assert result.stdout.count("infer_zero_shot_ablation.py") == 1
    assert "/xd/" not in result.stdout
    assert not output.exists()


def test_resume_checkpoint_and_non_overwriting_eval_retry(tmp_path):
    train = tmp_path / "train"
    for epoch in (0, 1):
        path = train / f"epoch{epoch}"
        path.mkdir(parents=True)
        (path / "train_state.pt").touch()
        (path / "lora_adapter").mkdir()
    assert latest_checkpoint(train) == (1, train / "epoch1")
    dataset_root = tmp_path / "B" / "ucf"
    (dataset_root / "eval").mkdir(parents=True)
    path, done = next_eval_dir(dataset_root, resume=True)
    assert path == dataset_root / "eval_retry_1" and not done
    with pytest.raises(FileExistsError):
        next_eval_dir(dataset_root, resume=False)


def test_summarizer_uses_only_completed_metrics_and_reports_all_comparisons(tmp_path):
    for group, auc, ap in (("B", 0.70, 0.40), ("C", 0.75, 0.46),
                           ("D", 0.78, 0.50), ("E", 0.81, 0.55)):
        for dataset, value in (("ucf", auc), ("xd", ap)):
            path = tmp_path / group / dataset / "eval" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "num_failed_videos": 0, "num_videos": 2,
                "global_standard_auc": value,
                "global_standard_ap": value,
                "global_causal_auc": value - 0.01,
                "global_causal_ap": value - 0.01,
                "mean_window_processing_ms": 12.0,
                "p95_window_processing_ms": 15.0,
                "rtf_processing": 0.2,
            }))
    rows = make_rows(tmp_path)
    assert [row["group"] for row in rows] == ["B", "C", "D", "E"]
    write_results(tmp_path, rows)
    with (tmp_path / "results.csv").open(newline="") as handle:
        assert [row["group"] for row in csv.DictReader(handle)] == ["B", "C", "D", "E"]
    report = (tmp_path / "experiment_report.md").read_text()
    for comparison in COMPARISONS:
        assert comparison in report
    assert "F-PESL" not in report


def test_ucf_only_summary_keeps_xd_metrics_empty(tmp_path):
    for group, auc in (("B", 0.70), ("C", 0.75)):
        path = tmp_path / group / "ucf" / "eval" / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "num_failed_videos": 0, "num_videos": 2,
            "global_standard_auc": auc, "global_standard_ap": 0.4,
            "mean_window_processing_ms": 12.0, "p95_window_processing_ms": 15.0,
            "rtf_processing": 0.2,
        }))
    rows = make_rows(tmp_path, ("ucf",))
    assert [row["group"] for row in rows] == ["B", "C"]
    assert all(row["xd_ap"] is None for row in rows)
    write_results(tmp_path, rows, ("ucf",))
    with (tmp_path / "results.csv").open(newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["xd_ap"] == ""
    assert "UCF AUC +0.05000" in (tmp_path / "experiment_report.md").read_text()
