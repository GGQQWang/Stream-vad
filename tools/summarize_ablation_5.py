"""Summarize existing A-E inference metrics without inventing missing results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GROUPS = ("A", "B", "C", "D", "E")
DATASETS = ("ucf", "xd")
FIELDS = (
    "group", "ucf_auc", "ucf_causal_auc", "xd_ap", "xd_causal_ap", "xd_auc",
    "ucf_mean_window_ms", "ucf_p95_window_ms", "ucf_rtf",
    "xd_mean_window_ms", "xd_p95_window_ms", "xd_rtf",
    "ucf_score_forward_ms", "xd_score_forward_ms",
    "ucf_generation_ms", "xd_generation_ms",
    "ucf_generation_failure_rate", "xd_generation_failure_rate",
)
COMPARISONS = {
    "C - B": ("C", "B", "semantic supervision without history"),
    "D - B": ("D", "B", "history modeling plus state-guided fusion with anomaly supervision"),
    "E - C": ("E", "C", "history modeling plus state-guided fusion with full supervision"),
    "E - D": ("E", "D", "semantic supervision with history"),
}


def latest_metrics(root: Path, group: str, dataset: str) -> dict | None:
    base = root / group / dataset
    paths = [base / "eval" / "metrics.json"]
    paths += sorted(base.glob("eval_retry_*/metrics.json"), key=lambda path: int(path.parent.name.rsplit("_", 1)[-1]))
    for path in reversed(paths):
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if int(metrics.get("num_failed_videos", 1)) != 0:
            continue
        if metrics.get("global_standard_auc") is None or int(metrics.get("num_videos", 0)) < 1:
            continue
        return metrics
    return None


def make_rows(root: Path, required_datasets: tuple[str, ...] = DATASETS) -> list[dict]:
    rows = []
    for group in GROUPS:
        ucf = latest_metrics(root, group, "ucf") if "ucf" in required_datasets else None
        xd = latest_metrics(root, group, "xd") if "xd" in required_datasets else None
        if any({"ucf": ucf, "xd": xd}[dataset] is None for dataset in required_datasets):
            continue
        ucf = ucf or {}
        xd = xd or {}
        row = {
            "group": group,
            "ucf_auc": ucf.get("global_standard_auc"),
            "ucf_causal_auc": ucf.get("global_causal_auc"),
            "xd_ap": xd.get("global_standard_ap"),
            "xd_causal_ap": xd.get("global_causal_ap"),
            "xd_auc": xd.get("global_standard_auc"),
            "ucf_mean_window_ms": ucf.get("mean_window_processing_ms", ucf.get("avg_window_ms")),
            "ucf_p95_window_ms": ucf.get("p95_window_processing_ms"),
            "ucf_rtf": ucf.get("rtf_processing"),
            "xd_mean_window_ms": xd.get("mean_window_processing_ms", xd.get("avg_window_ms")),
            "xd_p95_window_ms": xd.get("p95_window_processing_ms"),
            "xd_rtf": xd.get("rtf_processing"),
            "ucf_score_forward_ms": ucf.get("mean_score_forward_ms") if group != "A" else None,
            "xd_score_forward_ms": xd.get("mean_score_forward_ms") if group != "A" else None,
            "ucf_generation_ms": ucf.get("mean_generation_latency_ms") if group == "A" else None,
            "xd_generation_ms": xd.get("mean_generation_latency_ms") if group == "A" else None,
            "ucf_generation_failure_rate": ucf.get("generation_failure_rate") if group == "A" else None,
            "xd_generation_failure_rate": xd.get("generation_failure_rate") if group == "A" else None,
        }
        rows.append(row)
    return rows


def _format(value) -> str:
    return "" if value is None else f"{value:.5f}" if isinstance(value, float) else str(value)


def write_results(root: Path, rows: list[dict], required_datasets: tuple[str, ...] = DATASETS) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    columns = ("group",)
    if "ucf" in required_datasets:
        columns += ("ucf_auc", "ucf_mean_window_ms", "ucf_p95_window_ms", "ucf_rtf")
    if "xd" in required_datasets:
        columns += ("xd_ap", "xd_auc", "xd_mean_window_ms", "xd_p95_window_ms", "xd_rtf")
    lines = ["# A-E Ablation Results", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[key]) for key in columns) + " |")
    lines.extend(["", "Metrics come only from existing completed inference outputs; missing groups have no row.",
                  "Available UCF AUC and XD AP/AUC use the existing standard frame expansion; causal metrics are in results.csv.",
                  "A is a zero-shot generative baseline. Its generation latency is not directly comparable to score-head latency."])
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_group = {row["group"]: row for row in rows}
    report = ["# A-E Experiment Report", "",
              "B/C use the current window only. D/E use the causal SSM plus state-guided fusion.",
              "All B-E runs disable the IBQ future objective and use the final epoch checkpoint, not test-AUC selection.",
              "A uses deterministic raw-Qwen generation and must be compared separately on latency.", ""]
    for label, (positive, negative, explanation) in COMPARISONS.items():
        if positive not in by_group or negative not in by_group:
            report.append(f"- {label}: unavailable until both groups have complete selected-dataset evaluation.")
            continue
        left, right = by_group[positive], by_group[negative]
        deltas = []
        if left["ucf_auc"] is not None and right["ucf_auc"] is not None:
            deltas.append(f"UCF AUC {left['ucf_auc'] - right['ucf_auc']:+.5f}")
        if left["xd_ap"] is not None and right["xd_ap"] is not None:
            deltas.append(f"XD AP {left['xd_ap'] - right['xd_ap']:+.5f}")
        report.append(f"- {label}: {', '.join(deltas)}; {explanation}.")
    report.extend(["", "D-B and E-C vary the complete history path (SSM and state-guided fusion), not the SSM operator alone.",
                   "The standard score expansion assigns a window score to that window's frames. Strict emission-time causal metrics are recorded separately."])
    (root / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("both", *DATASETS), default="both")
    args = parser.parse_args()
    required_datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    rows = make_rows(args.output_root, required_datasets)
    write_results(args.output_root, rows, required_datasets)
    print(f"Wrote {len(rows)} completed A-E rows to {args.output_root}")


if __name__ == "__main__":
    main()
