"""Evaluate event-macro Stage-AP from Stage-1 <SCORE> hidden states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hivau_dataset import HIVAUDataset  # noqa: E402
from infer_stage1_ucf import (  # noqa: E402
    FRAMES_PER_CLIP,
    MAX_PIXELS,
    MAX_WINDOWS,
    MIN_PIXELS,
    SAMPLE_INTERVAL,
    SPATIAL_FUSION_MODES,
    infer_video,
    load_stage1_model,
    normalize_manifest,
)
from mil_utils import group_video_chunks  # noqa: E402
from stage_ap import aggregate_event_metrics, evaluate_video_stage_ap  # noqa: E402


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f} ({100.0 * value:.2f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline event-macro Stage-AP on UCF-Crime score-token representations.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--feature-cache-root", default="")
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage-ap-feature", choices=("score_hidden",), default="score_hidden")
    parser.add_argument("--debug-state", action="store_true")
    args = parser.parse_args()

    args.device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_manifest = normalize_manifest(args.test_manifest, output_dir, args.video_id)
    model, processor, tokenizer, dtype, prompt_text = load_stage1_model(args)
    dataset = HIVAUDataset(
        normalized_manifest,
        args.video_root,
        total_sampled_frames=FRAMES_PER_CLIP,
        sample_interval=SAMPLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        feature_cache_root=args.feature_cache_root or None,
        feature_cache_model_id=args.model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        require_spatial=(model.visual_fusion in SPATIAL_FUSION_MODES),
    )
    grouped = group_video_chunks(dataset.samples)
    if args.video_id:
        selected_video_id = Path(args.video_id).stem
        if selected_video_id not in grouped:
            raise ValueError(f"video_id={selected_video_id!r} not found in dataset")
        grouped = {selected_video_id: grouped[selected_video_id]}

    event_records: list[dict] = []
    failed_videos: list[dict] = []
    for video_id, refs in tqdm(grouped.items(), desc="Stage-AP videos"):
        try:
            result = infer_video(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                dataset=dataset,
                refs=refs,
                device=args.device,
                dtype=dtype,
                prompt_text=prompt_text,
                output_dir=output_dir,
                gt_root=args.gt_root,
                debug_state=args.debug_state,
                collect_score_hidden=True,
            )
            event_records.extend(evaluate_video_stage_ap(
                video_id=video_id,
                frame_gt=result["gt"],
                window_rows=result["window_rows"],
                score_hidden=result["score_hidden"].numpy(),
            ))
        except Exception as exc:
            failed_videos.append({"video_id": video_id, "error": str(exc)[:500]})
            print(f"FAILED video={video_id}: {exc}")

    summary = aggregate_event_metrics(event_records)
    summary.update({
        "num_videos": len(grouped) - len(failed_videos),
        "num_failed_videos": len(failed_videos),
        "failed_videos": failed_videos,
        "interval_convention": "frame_[start,end)_and_window_[start_frame,end_frame)",
        "ambiguous_window_rule": "exclude_windows_overlapping_multiple_anomaly_events",
    })
    with open(output_dir / "stage_ap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "stage_ap_per_event.json", "w") as f:
        json.dump(event_records, f, indent=2)

    print("Stage-AP Evaluation")
    print("-------------------")
    print(f"Valid events  : {summary['num_valid_events']}")
    print(f"Skipped events: {summary['num_skipped_events']}")
    print(f"Failed videos : {summary['num_failed_videos']}")
    print("")
    print(f"EM-AP    : {_format_metric(summary['em_ap'])}")
    print(f"ML-AP    : {_format_metric(summary['ml_ap'])}")
    print(f"EL-AP    : {_format_metric(summary['el_ap'])}")
    print(f"Stage-AP : {_format_metric(summary['stage_ap'])}")


if __name__ == "__main__":
    main()
