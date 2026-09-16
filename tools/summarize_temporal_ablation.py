"""Combine unchanged UCF AUC and Stage-AP outputs for temporal ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "method",
    "auc",
    "em_ap",
    "ml_ap",
    "el_ap",
    "stage_ap",
    "mean_latency_ms",
    "p95_latency_ms",
    "throughput_windows_s",
    "history_elements",
    "history_buffer_bytes",
    "history_buffer_mb",
    "temporal_params",
    "trainable_params",
    "peak_gpu_memory_gb",
)

FAIR_PROTOCOL_FIELDS = (
    "temporal_history",
    "visual_fusion",
    "inference_dtype",
    "test_manifest",
    "feature_cache_root",
    "profile_warmup_windows",
    "profile_seen_windows",
    "profile_measured_windows",
    "profiling_batch_size",
    "profiling_scope",
)


def validate_fair_protocol(output_root: str | Path, methods: list[str]) -> None:
    reference = None
    reference_method = None
    for method in methods:
        path = Path(output_root) / f"infer_ucf_temporal_{method}_epoch0" / "metrics.json"
        with open(path) as f:
            metrics = json.load(f)
        protocol = {field: metrics.get(field) for field in FAIR_PROTOCOL_FIELDS}
        missing = [field for field, value in protocol.items() if value is None]
        if missing:
            raise ValueError(f"{method}: missing profiling protocol fields {missing}")
        if reference is None:
            reference = protocol
            reference_method = method
        elif protocol != reference:
            differences = {
                field: (reference[field], protocol[field])
                for field in FAIR_PROTOCOL_FIELDS
                if reference[field] != protocol[field]
            }
            raise ValueError(
                f"profiling protocol mismatch: {reference_method} vs {method}: {differences}"
            )


def load_temporal_result(output_root: str | Path, method: str) -> dict:
    root = Path(output_root)
    inference_path = root / f"infer_ucf_temporal_{method}_epoch0" / "metrics.json"
    stage_path = root / f"stage_ap_ucf_temporal_{method}_epoch0" / "stage_ap_summary.json"
    if not inference_path.is_file():
        raise FileNotFoundError(f"missing inference metrics: {inference_path}")
    if not stage_path.is_file():
        raise FileNotFoundError(f"missing Stage-AP metrics: {stage_path}")
    with open(inference_path) as f:
        inference = json.load(f)
    with open(stage_path) as f:
        stage = json.load(f)
    saved_method = inference.get("temporal_model")
    if saved_method != method:
        raise ValueError(
            f"temporal method mismatch in {inference_path}: saved={saved_method!r}, expected={method!r}"
        )
    if int(inference.get("profiling_batch_size", 0)) != 1:
        raise ValueError(f"{method}: metrics were not produced by batch-1 online profiling")
    if int(inference.get("num_failed_videos", 0)) != 0:
        raise ValueError(f"{method}: inference contains failed videos")
    if int(stage.get("num_failed_videos", 0)) != 0:
        raise ValueError(f"{method}: Stage-AP evaluation contains failed videos")
    if int(inference.get("profile_measured_windows", 0)) < 1:
        raise ValueError(f"{method}: online profiling contains no measured windows")
    if stage.get("aggregation") != "event_macro" or stage.get("feature") != "score_hidden":
        raise ValueError(f"{method}: unexpected Stage-AP protocol metadata")
    stage_values = [stage.get(key) for key in ("em_ap", "ml_ap", "el_ap")]
    if all(value is not None for value in stage_values):
        expected_stage_ap = sum(float(value) for value in stage_values) / 3.0
        if abs(float(stage.get("stage_ap")) - expected_stage_ap) > 1e-10:
            raise ValueError(f"{method}: Stage-AP is inconsistent with EM/ML/EL macro AP")
    return {
        "method": method,
        "auc": inference.get("global_standard_auc"),
        "em_ap": stage.get("em_ap"),
        "ml_ap": stage.get("ml_ap"),
        "el_ap": stage.get("el_ap"),
        "stage_ap": stage.get("stage_ap"),
        "mean_latency_ms": inference.get("mean_latency_ms"),
        "p95_latency_ms": inference.get("p95_latency_ms"),
        "throughput_windows_s": inference.get("throughput_windows_s"),
        "history_elements": inference.get("history_elements"),
        "history_buffer_bytes": inference.get("history_buffer_bytes"),
        "history_buffer_mb": inference.get("history_buffer_mb"),
        "temporal_params": inference.get("temporal_params"),
        "trainable_params": inference.get("trainable_params"),
        "peak_gpu_memory_gb": inference.get("peak_gpu_memory_gb"),
    }


def write_temporal_results(rows: list[dict], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "temporal_ablation_results.csv"
    json_path = output_dir / "temporal_ablation_results.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def print_results(rows: list[dict]) -> None:
    print(
        "Method\tAUC\tEM-AP\tML-AP\tEL-AP\tStage-AP\tMean ms\tP95 ms\t"
        "Win/s\tBuffer MB\tTemporal Params\tTrainable Params\tPeak GB"
    )
    for row in rows:
        print("\t".join([
            row["method"],
            _fmt(row["auc"]),
            _fmt(row["em_ap"]),
            _fmt(row["ml_ap"]),
            _fmt(row["el_ap"]),
            _fmt(row["stage_ap"]),
            _fmt(row["mean_latency_ms"], 3),
            _fmt(row["p95_latency_ms"], 3),
            _fmt(row["throughput_windows_s"], 3),
            _fmt(row["history_buffer_mb"], 6),
            _fmt(row["temporal_params"], 0),
            _fmt(row["trainable_params"], 0),
            _fmt(row["peak_gpu_memory_gb"], 3),
        ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/data3/wgq/outputs")
    parser.add_argument("--results-dir", default="/data3/wgq/outputs")
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    validate_fair_protocol(args.output_root, args.methods)
    rows = [load_temporal_result(args.output_root, method) for method in args.methods]
    csv_path, json_path = write_temporal_results(rows, args.results_dir)
    print_results(rows)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
