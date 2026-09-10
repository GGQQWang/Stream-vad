"""Summarize LVLM inference ablation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = [
    ("score-only", "Score-only"),
    ("parallel-16", "Parallel-16"),
    ("ar-16", "AR-16"),
    ("ar-64", "AR-64"),
]
LEGACY_MODE_ALIASES = {
    "score-only": "forward",
    "ar-16": "ar16",
    "ar-64": "ar64",
}


def load_metrics(root: Path, mode: str) -> dict | None:
    path = root / mode / "metrics.json"
    if not path.is_file() and mode in LEGACY_MODE_ALIASES:
        path = root / LEGACY_MODE_ALIASES[mode] / "metrics.json"
    if not path.is_file():
        return None
    with open(path, "r") as f:
        return json.load(f)


def fmt(value, suffix: str = "") -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{value:.4f}{suffix}"
    return str(value)


def pct_change(value, base) -> float | None:
    if value is None or base in (None, 0):
        return None
    return 100.0 * (float(value) - float(base)) / float(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data3/wgq/outputs/lvlm_inference_ablation")
    args = parser.parse_args()
    root = Path(args.output_dir)
    metrics = {mode: load_metrics(root, mode) for mode, _ in ROWS}

    print("| LVLM Inference | AUC | Mean Lat. | P95 Lat. | Throughput | RTF | Peak Mem |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for mode, label in ROWS:
        m = metrics[mode] or {}
        auc = m.get("global_standard_auc")
        peak_mem = m.get("peak_memory_allocated_gb")
        print(
            f"| {label} | {fmt(auc)} | {fmt(m.get('latency_mean_ms'))} ms | "
            f"{fmt(m.get('latency_p95_ms'))} ms | {fmt(m.get('throughput_windows_per_sec'))} | "
            f"{fmt(m.get('RTF'))} | {fmt(peak_mem)} GB |"
        )

    base = metrics.get("score-only") or {}
    print()
    print("| LVLM Inference | Latency Increase | Throughput Drop | Memory Increase |")
    print("|---|---:|---:|---:|")
    for mode, label in ROWS[1:]:
        m = metrics[mode] or {}
        latency_inc = pct_change(m.get("latency_mean_ms"), base.get("latency_mean_ms"))
        throughput_change = pct_change(m.get("throughput_windows_per_sec"), base.get("throughput_windows_per_sec"))
        throughput_drop = -throughput_change if throughput_change is not None else None
        memory_inc = pct_change(m.get("peak_memory_allocated_gb"), base.get("peak_memory_allocated_gb"))
        print(f"| {label} | {fmt(latency_inc, '%')} | {fmt(throughput_drop, '%')} | {fmt(memory_inc, '%')} |")


if __name__ == "__main__":
    main()
