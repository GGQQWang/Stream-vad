"""Validation and result aggregation for long-horizon temporal scaling."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


TEMPORAL_METHODS = ("ssm", "lstm", "stc", "qformer", "transformer")
HISTORY_HORIZONS = (4, 8, 16, 32, 64)
MAX_TRAINING_HORIZON = 64
RESULT_FIELDS = (
    "method",
    "history_horizon",
    "auc",
    "ap",
    "mean_latency_ms",
    "p95_latency_ms",
    "throughput_windows_s",
    "history_buffer_bytes",
    "history_buffer_mb",
    "peak_gpu_memory_gb",
    "v_normal_k16",
    "v_anomaly_k16",
    "v_boundary_k16",
)


def validate_long64_checkpoint(
    state: dict,
    method: str,
    *,
    max_horizon: int = MAX_TRAINING_HORIZON,
) -> None:
    """Fail if a checkpoint was not trained under the common long64 protocol."""
    if method not in TEMPORAL_METHODS:
        raise ValueError(f"unknown temporal method {method!r}")
    saved_method = str(state.get("temporal_model", ""))
    if saved_method != method:
        raise ValueError(
            f"checkpoint temporal_model={saved_method!r}, expected {method!r}"
        )
    required = {
        "max_training_horizon": max_horizon,
        "max_windows": max_horizon,
        "temporal_history": max_horizon,
    }
    for key, expected in required.items():
        if key not in state:
            raise ValueError(f"long64 checkpoint is missing {key}")
        try:
            actual = int(state[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"long64 checkpoint has invalid {key}={state[key]!r}"
            ) from exc
        if actual != int(expected):
            raise ValueError(
                f"long64 checkpoint {key}={state[key]!r}, expected {expected}"
            )
    if state.get("training_history_reset") != "chunk":
        raise ValueError(
            "long64 checkpoint must use chunk-boundary temporal reset to cap training history"
        )
    config = state.get("temporal_config")
    if not isinstance(config, dict) or config.get("name") != method:
        raise ValueError("long64 checkpoint has invalid temporal_config")
    if int(state.get("temporal_state_dim", state.get("d_ssm", 0))) <= 0:
        raise ValueError("long64 checkpoint has invalid temporal state dimension")
    if method in {"stc", "qformer", "transformer"}:
        if int(config.get("history_length", 0)) != int(max_horizon):
            raise ValueError(
                f"{method} checkpoint history capacity is not {max_horizon}"
            )


def _read_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing long-horizon metrics: {path}")
    with open(path) as handle:
        return json.load(handle)


def collect_scaling_results(
    results_root: str | Path,
    methods: tuple[str, ...] | list[str] = TEMPORAL_METHODS,
    horizons: tuple[int, ...] | list[int] = HISTORY_HORIZONS,
) -> list[dict]:
    root = Path(results_root)
    rows = []
    reference_protocol = None
    for method in methods:
        method_rows = []
        for horizon in horizons:
            metrics = _read_metrics(root / method / f"T{int(horizon)}" / "metrics.json")
            if metrics.get("temporal_model") != method:
                raise ValueError(f"{method}/T{horizon}: temporal model mismatch")
            if int(metrics.get("history_horizon", 0)) != int(horizon):
                raise ValueError(f"{method}/T{horizon}: history horizon mismatch")
            if int(metrics.get("max_training_horizon", 0)) != MAX_TRAINING_HORIZON:
                raise ValueError(f"{method}/T{horizon}: checkpoint is not long64")
            if int(metrics.get("num_failed_videos", -1)) != 0:
                raise ValueError(f"{method}/T{horizon}: evaluation contains failed videos")
            protocol = {
                key: metrics.get(key)
                for key in (
                    "visual_fusion",
                    "inference_dtype",
                    "test_manifest",
                    "feature_cache_root",
                    "profile_warmup_windows",
                    "profiling_batch_size",
                    "profiling_scope",
                    "effectiveness_protocol",
                    "representation_k_eval",
                )
            }
            if any(value is None for value in protocol.values()):
                raise ValueError(f"{method}/T{horizon}: incomplete protocol metadata")
            if reference_protocol is None:
                reference_protocol = protocol
            elif protocol != reference_protocol:
                raise ValueError(
                    f"{method}/T{horizon}: protocol mismatch: "
                    f"{protocol} != {reference_protocol}"
                )
            if horizon < 16 and any(
                metrics.get(key) is not None
                for key in ("v_normal_k16", "v_anomaly_k16", "v_boundary_k16")
            ):
                raise ValueError(f"{method}/T{horizon}: fixed-k16 metrics must be null")
            row = {field: metrics.get(field) for field in RESULT_FIELDS}
            method_rows.append(row)
            rows.append(row)

        buffer_sizes = [int(row["history_buffer_bytes"]) for row in method_rows]
        if method in {"ssm", "lstm"}:
            if len(set(buffer_sizes)) != 1:
                raise ValueError(f"{method}: native recurrent buffer changed with T")
        elif any(right <= left for left, right in zip(buffer_sizes, buffer_sizes[1:])):
            raise ValueError(f"{method}: explicit history buffer did not grow with T")
    return rows


def write_scaling_results(rows: list[dict], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "long_horizon_scaling.csv"
    json_path = output_dir / "long_horizon_scaling.json"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as handle:
        json.dump({
            "definitions": {
                "history_horizon": "maximum model-visible temporal windows",
                "effectiveness": "strict sliding recent-T history",
                "recurrent_efficiency": "native O(1) streaming state without replay",
                "explicit_efficiency": "native streaming with a T-window retained buffer",
                "representation_k_eval": 16,
            },
            "rows": rows,
        }, handle, indent=2)
    return csv_path, json_path


def write_scaling_figures(rows: list[dict], output_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    specs = (
        ("auc", "AUC", "long_horizon_auc.png", HISTORY_HORIZONS),
        ("mean_latency_ms", "Native Mean Latency (ms/window)", "long_horizon_latency.png", HISTORY_HORIZONS),
        ("history_buffer_mb", "Native History Buffer (MB)", "long_horizon_buffer.png", HISTORY_HORIZONS),
        ("v_boundary_k16", "V_B@16", "long_horizon_v_boundary_k16.png", (16, 32, 64)),
    )
    for field, ylabel, filename, allowed_horizons in specs:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for method in TEMPORAL_METHODS:
            selected = [
                row for row in rows
                if row["method"] == method
                and int(row["history_horizon"]) in allowed_horizons
                and row[field] is not None
                and math.isfinite(float(row[field]))
            ]
            selected.sort(key=lambda row: int(row["history_horizon"]))
            axis.plot(
                [int(row["history_horizon"]) for row in selected],
                [float(row[field]) for row in selected],
                marker="o",
                label=method.upper() if method == "ssm" else method.title(),
            )
        axis.set_xlabel("History Horizon T (windows)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(allowed_horizons))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=200)
        plt.close(figure)
        figures.append(path)
    return figures
