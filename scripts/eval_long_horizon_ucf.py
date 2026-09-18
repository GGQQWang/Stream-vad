"""Evaluate one long64 checkpoint at one strict history horizon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
    OnlineInferenceProfiler,
    _auc_ap,
    infer_video,
    load_stage1_model,
    normalize_manifest,
    profile_cached_video,
)
from long_horizon_scaling import (  # noqa: E402
    HISTORY_HORIZONS,
    MAX_TRAINING_HORIZON,
    TEMPORAL_METHODS,
    validate_long64_checkpoint,
)
from mil_utils import group_video_chunks  # noqa: E402
from temporal_representation import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    evaluate_representation_cache,
    save_json,
    save_repr_video,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=TEMPORAL_METHODS, required=True)
    parser.add_argument("--history-horizon", type=int, choices=HISTORY_HORIZONS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-warmup-windows", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.profile_warmup_windows < 0:
        raise ValueError("--profile-warmup-windows must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.stage1_dir) / "train_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    validate_long64_checkpoint(state, args.method)

    args.device = torch.device(args.device)
    args.temporal_model = args.method
    args.temporal_history = 0
    normalized_manifest = normalize_manifest(args.test_manifest, output_dir, "")
    model, processor, tokenizer, dtype, prompt_text = load_stage1_model(args)
    if model.temporal_history != MAX_TRAINING_HORIZON:
        raise ValueError("loaded model does not expose long64 temporal capacity")
    dataset = HIVAUDataset(
        normalized_manifest,
        args.video_root,
        total_sampled_frames=FRAMES_PER_CLIP,
        sample_interval=SAMPLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        feature_cache_root=args.feature_cache_root,
        feature_cache_model_id=args.model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        require_spatial=(model.visual_fusion in SPATIAL_FUSION_MODES),
    )
    grouped = group_video_chunks(dataset.samples)

    profiler = OnlineInferenceProfiler(args.profile_warmup_windows)
    for refs in tqdm(grouped.values(), desc=f"Profile {args.method} T={args.history_horizon}"):
        profile_cached_video(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            refs=refs,
            device=args.device,
            dtype=dtype,
            prompt_text=prompt_text,
            profiler=profiler,
            history_horizon=args.history_horizon,
        )
    profile_metrics = profiler.summary(args.device)

    representation_dir = output_dir / "representation"
    representation_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    all_gt = []
    all_scores = []
    videos = []
    for video_id, refs in tqdm(
        grouped.items(), desc=f"Strict effectiveness {args.method} T={args.history_horizon}",
    ):
        with torch.inference_mode():
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
                debug_state=False,
                collect_score_hidden=True,
                write_outputs=False,
                history_horizon=args.history_horizon,
            )
        rows = result["window_rows"]
        hidden = result["score_hidden"].numpy().astype(np.float32, copy=False)
        cache_path = representation_dir / f"{video_id}.npz"
        save_repr_video(
            cache_path,
            method=args.method,
            video_id=video_id,
            n_frames=result["n_frames"],
            fps=result["fps"],
            window_index=np.asarray([row["window_index"] for row in rows], dtype=np.int64),
            start_frame=np.asarray([row["start_frame"] for row in rows], dtype=np.int64),
            valid_end_frame=np.asarray([row["end_frame"] for row in rows], dtype=np.int64),
            score_hidden=hidden,
            storage_dtype="float16",
        )
        entries.append({
            "video_id": video_id,
            "file": cache_path.name,
            "num_windows": len(rows),
            "hidden_dim": int(hidden.shape[1]),
        })
        all_gt.append(result["gt"])
        all_scores.append(result["standard_scores"])
        videos.append({
            "video_id": video_id,
            "n_frames": result["n_frames"],
            "num_windows": result["num_windows"],
            "standard_auc": result["standard_auc"],
            "standard_ap": result["standard_ap"],
        })

    entries.sort(key=lambda item: item["video_id"])
    save_json(representation_dir / "manifest.json", {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "method": args.method,
        "history_horizon": args.history_horizon,
        "stage1_dir": str(Path(args.stage1_dir).resolve()),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "feature_cache_root": str(Path(args.feature_cache_root).resolve()),
        "storage_dtype": "float16",
        "representation": "final_layer_SCORE_hidden_input_to_score_head",
        "interval": "frame_[start_frame,valid_end_frame)",
        "num_videos": len(entries),
        "videos": entries,
    })

    v_normal = v_anomaly = v_boundary = None
    if args.history_horizon >= 16:
        representation = evaluate_representation_cache(
            representation_dir, args.gt_root, k_values=(16,),
        )[0]
        v_normal = representation["v_normal"]
        v_anomaly = representation["v_anomaly"]
        v_boundary = representation["v_boundary"]

    gt = np.concatenate(all_gt) if all_gt else np.empty(0, dtype=np.int64)
    scores = np.concatenate(all_scores) if all_scores else np.empty(0, dtype=np.float32)
    auc, ap = _auc_ap(scores, gt)
    metrics = {
        "method": args.method,
        "temporal_model": model.temporal_model,
        "history_horizon": args.history_horizon,
        "max_training_horizon": int(state["max_training_horizon"]),
        "max_windows": int(state["max_windows"]),
        "training_history_reset": state["training_history_reset"],
        "temporal_history": model.temporal_history,
        "temporal_config": state["temporal_config"],
        "visual_fusion": model.visual_fusion,
        "inference_dtype": str(dtype),
        "test_manifest": str(args.test_manifest),
        "feature_cache_root": str(args.feature_cache_root),
        "num_videos": len(videos),
        "num_failed_videos": 0,
        "auc": auc,
        "ap": ap,
        "global_standard_auc": auc,
        "global_standard_ap": ap,
        "effectiveness_protocol": "strict_sliding_recent_T_temporal_only_replay",
        "native_efficiency_protocol": (
            "native_recurrent_state" if args.method in {"ssm", "lstm"}
            else "native_explicit_T_window_buffer"
        ),
        "representation_k_eval": 16,
        "v_normal_k16": v_normal,
        "v_anomaly_k16": v_anomaly,
        "v_boundary_k16": v_boundary,
        "temporal_params": model.temporal_param_count,
        "trainable_params": model.training_trainable_param_count,
        **profile_metrics,
        "videos": videos,
    }
    save_json(output_dir / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
