"""Fully offline FDR@k and Temporal Variation@k evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from temporal_representation import (  # noqa: E402
    DEFAULT_K_VALUES,
    evaluate_representation_cache,
    save_json,
)


TEMPORAL_MODELS = ("ssm", "lstm", "stc", "qformer", "transformer")


LONG_FIELDS = (
    "method", "k", "fdr", "normal_scatter", "anomaly_scatter", "between_scatter",
    "v_normal", "v_anomaly", "v_boundary", "v_normal_micro", "v_anomaly_micro",
    "v_boundary_micro", "fdr_valid_videos", "normal_block_count", "anomaly_block_count",
    "fdr_normal_block_count_used", "fdr_anomaly_block_count_used", "num_normal_segments",
    "num_anomaly_events", "num_boundaries", "normal_pair_count", "anomaly_pair_count",
    "boundary_pair_count", "fdr_skip_reasons", "fdr_direction", "variation_aggregation",
    "fdr_skipped_videos",
)


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _load_common_cache_metadata(analysis_root: Path, methods: list[str]) -> dict:
    common = None
    for method in methods:
        path = analysis_root / method / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing representation manifest: {path}")
        with open(path) as handle:
            manifest = json.load(handle)
        if manifest.get("method") != method or manifest.get("status") != "complete":
            raise ValueError(f"invalid or incomplete representation manifest: {path}")
        protocol = {
            "schema_version": manifest.get("schema_version"),
            "test_manifest": manifest.get("test_manifest"),
            "feature_cache_root": manifest.get("feature_cache_root"),
            "storage_dtype": manifest.get("storage_dtype"),
            "representation": manifest.get("representation"),
            "interval": manifest.get("interval"),
            "num_videos": manifest.get("num_videos"),
        }
        if common is None:
            common = protocol
        elif protocol != common:
            raise ValueError(f"representation protocol mismatch for method={method}: {protocol} != {common}")
    return common or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root", default="/data3/wgq/outputs/temporal_repr_analysis",
    )
    parser.add_argument("--gt-root", default="/data1/wjq/data/UCF_Crime/testing/gt_labels")
    parser.add_argument("--output-dir", default="/data3/wgq/outputs/temporal_repr_analysis")
    parser.add_argument(
        "--methods", nargs="+", choices=TEMPORAL_MODELS,
        default=["ssm", "lstm", "stc", "qformer", "transformer"],
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")

    analysis_root = Path(args.analysis_root)
    output_dir = Path(args.output_dir)
    common_protocol = _load_common_cache_metadata(analysis_root, args.methods)
    rows = []
    for method in args.methods:
        rows.extend(evaluate_representation_cache(
            analysis_root / method,
            args.gt_root,
            args.k_values,
        ))
    rows.sort(key=lambda row: (args.methods.index(row["method"]), int(row["k"])))

    write_csv(output_dir / "temporal_representation_metrics.csv", rows, LONG_FIELDS)
    save_json(output_dir / "temporal_representation_metrics.json", {
        "definitions": {
            "representation": "L2-normalized final-layer SCORE hidden input to score_head",
            "fdr": "between_scatter/(normal_scatter+anomaly_scatter+1e-12)",
            "fdr_aggregation": "per-video then video macro",
            "variation": "1-cosine(z_t,z_t+k)",
            "variation_aggregation": "pair mean within group then group macro",
            "mixed_windows": "excluded from blocks and pair endpoints",
            "json_missing_value": "null; CSV uses nan",
        },
        "cache_protocol": common_protocol,
        "k_values": [int(value) for value in args.k_values],
        "rows": rows,
    })

    k16_rows = []
    for row in rows:
        if int(row["k"]) == 16:
            k16_rows.append({
                "method": row["method"],
                "fdr_k16": row["fdr"],
                "v_normal_k16": row["v_normal"],
                "v_anomaly_k16": row["v_anomaly"],
                "v_boundary_k16": row["v_boundary"],
                "fdr_valid_videos": row["fdr_valid_videos"],
                "normal_block_count": row["normal_block_count"],
                "anomaly_block_count": row["anomaly_block_count"],
                "num_normal_segments": row["num_normal_segments"],
                "num_anomaly_events": row["num_anomaly_events"],
                "num_boundaries": row["num_boundaries"],
                "normal_pair_count": row["normal_pair_count"],
                "anomaly_pair_count": row["anomaly_pair_count"],
                "boundary_pair_count": row["boundary_pair_count"],
            })
    write_csv(
        output_dir / "temporal_representation_k16.csv",
        k16_rows,
        (
            "method", "fdr_k16", "v_normal_k16", "v_anomaly_k16", "v_boundary_k16",
            "fdr_valid_videos", "normal_block_count", "anomaly_block_count",
            "num_normal_segments", "num_anomaly_events", "num_boundaries",
            "normal_pair_count", "anomaly_pair_count", "boundary_pair_count",
        ),
    )

    write_csv(
        output_dir / "temporal_representation_fdr_curve.csv",
        rows,
        ("method", "k", "fdr", "fdr_valid_videos", "normal_block_count", "anomaly_block_count"),
    )
    for name, value, count in (
        ("normal", "v_normal", "normal_pair_count"),
        ("anomaly", "v_anomaly", "anomaly_pair_count"),
        ("boundary", "v_boundary", "boundary_pair_count"),
    ):
        write_csv(
            output_dir / f"temporal_representation_v_{name}_curve.csv",
            rows,
            ("method", "k", value, count),
        )
    print(f"Saved temporal representation metrics to {output_dir}")


if __name__ == "__main__":
    main()
