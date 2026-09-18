"""Aggregate 5x5 long-horizon evaluations and render paper figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from long_horizon_scaling import (  # noqa: E402
    HISTORY_HORIZONS,
    TEMPORAL_METHODS,
    collect_scaling_results,
    write_scaling_figures,
    write_scaling_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="/data3/wgq/outputs/long_horizon_scaling")
    parser.add_argument("--output-dir", default="/data3/wgq/outputs/long_horizon_scaling")
    parser.add_argument("--methods", nargs="+", choices=TEMPORAL_METHODS, default=list(TEMPORAL_METHODS))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HISTORY_HORIZONS, default=list(HISTORY_HORIZONS))
    args = parser.parse_args()
    rows = collect_scaling_results(args.results_root, args.methods, args.horizons)
    csv_path, json_path = write_scaling_results(rows, args.output_dir)
    figures = write_scaling_figures(rows, args.output_dir)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    for path in figures:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
