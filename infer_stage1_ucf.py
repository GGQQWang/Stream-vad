"""Strict streaming Stage-1 score_token inference on UCF-Crime test videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from hivau_dataset import HIVAUDataset, hivau_collate
from mil_utils import group_video_chunks
from pipeline_stage1 import (
    StreamingVADGenerationModel,
    _find_embed,
)


FRAMES_PER_CLIP = 16
SAMPLE_INTERVAL = 3
MAX_WINDOWS = 8
MIN_PIXELS = 100352
MAX_PIXELS = 100352


def _load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def _video_id_from_entry(entry: dict) -> str:
    for key in ("video_id", "id", "name"):
        if key in entry:
            return Path(str(entry[key])).stem
    for key in ("video", "video_path", "path", "filename"):
        if key in entry:
            return Path(str(entry[key])).stem
    raise ValueError(f"manifest entry does not contain a video id: {entry}")


def normalize_manifest(manifest_path: str | Path, output_dir: Path, video_id: str = "") -> Path:
    raw = _load_json(manifest_path)
    normalized: Dict[str, dict] = {}
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = ((_video_id_from_entry(item), item) for item in raw)
    else:
        raise ValueError("test manifest must be a dict or a list of dicts")

    wanted_video_id = Path(video_id).stem if video_id else ""
    for vid, meta in iterable:
        vid = Path(str(vid)).stem
        if wanted_video_id and vid != wanted_video_id:
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"manifest metadata must be a dict: video={vid}")
        n_frames = meta.get("n_frames", meta.get("num_frames"))
        fps = meta.get("fps")
        if n_frames is None or fps is None:
            raise ValueError(f"manifest must provide n_frames and fps: video={vid}")
        normalized[vid] = {
            **meta,
            "n_frames": int(n_frames),
            "fps": float(fps),
            "label": meta.get("label", []),
            "events": meta.get("events", []),
            "clips": meta.get("clips", []),
            "clips_caption": meta.get("clips_caption", []),
        }
    if not normalized:
        raise ValueError(f"no videos selected from manifest: video_id={video_id!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "_ucf_infer_manifest.normalized.json"
    with open(path, "w") as f:
        json.dump(normalized, f)
    return path


def _check_stage1_config(state: dict, model_path: str) -> None:
    expected = {
        "frames_per_clip": FRAMES_PER_CLIP,
        "sample_interval": SAMPLE_INTERVAL,
        "max_windows": MAX_WINDOWS,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }
    for key, value in expected.items():
        if key in state and int(state[key]) != int(value):
            raise ValueError(f"stage1 checkpoint {key}={state[key]!r}, expected {value}")
    if "feature_cache_model_id" in state and str(state["feature_cache_model_id"]) != str(model_path):
        raise ValueError(
            f"stage1 feature_cache_model_id={state['feature_cache_model_id']!r}, "
            f"expected model_path={model_path!r}"
        )
    objective = state.get("objective")
    if objective is not None and objective != "score_token":
        raise ValueError(f"Stage-1 inference requires objective='score_token', got {objective!r}")


def load_stage1_model(args) -> tuple[StreamingVADGenerationModel, object, object, torch.dtype, str]:
    stage1_dir = Path(args.stage1_dir)
    state_path = stage1_dir / "train_state.pt"
    lora_dir = stage1_dir / "lora_adapter"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing train_state.pt: {state_path}")
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"missing LoRA adapter directory: {lora_dir}")

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    _check_stage1_config(state, args.model_path)

    dtype = torch.bfloat16
    print("Loading Qwen2-VL base model ...")
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(args.device)
    qwen.config.use_cache = False
    qwen = PeftModel.from_pretrained(qwen, str(lora_dir), is_trainable=False).to(args.device)

    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = StreamingVADGenerationModel(
        qwen,
        d_ssm=int(state.get("d_ssm", 256)),
        llm_hidden=qwen.config.hidden_size,
        vit_micro_batch=1,
    ).to(args.device)
    model.ssm.load_state_dict(state["ssm"])
    model.adapter.load_state_dict(state["adapter"])
    model.score_head.load_state_dict(state["score_head"])
    model.score_query.data.copy_(state["score_query"].to(model.score_query.device, model.score_query.dtype))
    if "summary_query" in state:
        model.summary_query.data.copy_(state["summary_query"].to(model.summary_query.device, model.summary_query.dtype))
    if "alpha_logit" in state:
        model.alpha_logit.data.copy_(state["alpha_logit"].to(model.alpha_logit.device, model.alpha_logit.dtype))
    else:
        print("WARNING: alpha_logit missing from checkpoint; using model default.")
    model.debug_state = False
    model.eval()
    prompt_text = str(state.get("prompt", "Current video status:"))
    return model, processor, tokenizer, dtype, prompt_text


def _auc_ap(scores: np.ndarray, gt: np.ndarray) -> tuple[float | None, float | None]:
    if np.unique(gt).size < 2:
        return None, None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        raise ImportError("scikit-learn is required for UCF evaluation")
    return float(roc_auc_score(gt, scores)), float(average_precision_score(gt, scores))


def load_gt(gt_root: str | Path, video_id: str, n_frames: int) -> np.ndarray:
    path = Path(gt_root) / f"{video_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing GT file: {path}")
    gt = np.loadtxt(path, dtype=np.int64)
    gt = np.atleast_1d(gt).astype(np.int64)
    if len(gt) != int(n_frames):
        raise ValueError(f"{video_id}: GT length={len(gt)}, expected n_frames={n_frames}")
    return gt


def save_window_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "video_id",
        "window_index",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "score_logit",
        "score_prob",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _synchronize(device: torch.device) -> None:
    """Block until all pending work on ``device`` finishes.

    CUDA launches are asynchronous, so wall-clock timing without a sync only
    measures launch overhead, not actual execution.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_video(
    *,
    model: StreamingVADGenerationModel,
    processor,
    tokenizer,
    dataset: HIVAUDataset,
    refs,
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    output_dir: Path,
    gt_root: str | Path,
    debug_state: bool,
) -> dict:
    video_id = refs[0].video_id
    first_meta = dataset.samples[refs[0].index]
    n_frames = int(first_meta["n_frames"])
    fps = float(first_meta["fps"])
    gt = load_gt(gt_root, video_id, n_frames)
    embed_fn = _find_embed(model.qwen)
    ssm_cache: dict = {}
    rows: List[dict] = []

    # timing accumulators (wall-clock; processing excludes data loading)
    data_loading_sec = 0.0
    processing_sec = 0.0
    steady_processing_sec = 0.0   # excludes first chunk (CUDA warmup)
    first_chunk_scored = 0

    with torch.no_grad():
        for chunk_i, ref in enumerate(refs):
            # data loading (frame decode / feature-cache read) is timed
            # separately from model processing
            t0 = time.perf_counter()
            batch = hivau_collate([dataset[ref.index]])
            data_loading_sec += time.perf_counter() - t0

            valid_mask_cpu = batch["valid_mask"]
            valid_mask = valid_mask_cpu.to(device)
            if debug_state:
                print(
                    f"SSM_STATE video={video_id} chunk={chunk_i} "
                    f"reuse_prev={video_id in ssm_cache}"
                )

            t1 = time.perf_counter()

            if "features" in batch:
                window_batch = batch["features"].to(device=device, dtype=dtype)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                    state_emb, _, _, ssm_cache = model.encode_window_features(
                        window_batch,
                        valid_mask,
                        batch["video_id"],
                        ssm_cache,
                        training=False,
                    )
            else:
                frames = batch["frames"][0]
                valid_w_cpu = valid_mask_cpu[0].nonzero(as_tuple=True)[0]
                clips = [frames[int(w)] for w in valid_w_cpu.tolist()]
                processed = processor.image_processor(images=None, videos=clips, return_tensors="pt")
                pixel_values = processed["pixel_values_videos"].to(device)
                grid_thw = processed["video_grid_thw"].to(device)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                    state_emb, _, ssm_cache, _ = model.encode_stream(
                        pixel_values,
                        grid_thw,
                        valid_mask,
                        batch["video_id"],
                        ssm_cache,
                        training=False,
                    )

            valid_b, valid_w = valid_mask.nonzero(as_tuple=True)
            if len(valid_b) == 0:
                continue
            if chunk_i == 0:
                first_chunk_scored = len(valid_b)
            states = state_emb[valid_b, valid_w]
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                logits = model.forward_score_token(
                    states,
                    embed_fn,
                    tokenizer,
                    prompt_text=prompt_text,
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"{video_id}: non-finite score logits in chunk {chunk_i} "
                    f"(logits min={float(logits.min())}, max={float(logits.max())})"
                )
            probs = torch.sigmoid(logits).detach().float().cpu()
            logits_cpu = logits.detach().float().cpu()
            _synchronize(device)
            elapsed = time.perf_counter() - t1
            processing_sec += elapsed
            if chunk_i > 0:
                steady_processing_sec += elapsed
            start_frames = batch["window_start_frames"]
            valid_end_frames = batch["valid_end_frames"]
            for i, (b_t, w_t) in enumerate(zip(valid_b.cpu(), valid_w.cpu())):
                b = int(b_t.item())
                w = int(w_t.item())
                start = int(start_frames[b, w].item())
                end = int(valid_end_frames[b, w].item())
                rows.append({
                    "video_id": video_id,
                    "window_index": int(batch["chunk_start"][b]) + w,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": start / fps,
                    "end_sec": end / fps,
                    "score_logit": float(logits_cpu[i].item()),
                    "score_prob": float(probs[i].item()),
                })

    if video_id in ssm_cache:
        ssm_cache.pop(video_id, None)
    if debug_state:
        print(f"SSM_STATE_CLEAR video={video_id}")

    rows.sort(key=lambda r: int(r["window_index"]))
    if not rows:
        raise ValueError(f"{video_id}: no valid windows were produced")
    if int(rows[0]["start_frame"]) != 0:
        raise ValueError(f"{video_id}: first window starts at {rows[0]['start_frame']}, expected 0")
    for prev, cur in zip(rows[:-1], rows[1:]):
        if int(prev["end_frame"]) != int(cur["start_frame"]):
            raise ValueError(
                f"{video_id}: non-contiguous windows between "
                f"window {prev['window_index']} [{prev['start_frame']}, {prev['end_frame']}) "
                f"and window {cur['window_index']} [{cur['start_frame']}, {cur['end_frame']})"
            )
    if int(rows[-1]["end_frame"]) != n_frames:
        raise ValueError(f"{video_id}: last window ends at {rows[-1]['end_frame']}, expected n_frames={n_frames}")
    save_window_csv(output_dir / f"{video_id}_window_scores.csv", rows)

    standard_scores = np.zeros(n_frames, dtype=np.float32)
    causal_scores = np.full(n_frames, np.nan, dtype=np.float32)
    causal_valid = np.zeros(n_frames, dtype=bool)
    for row in rows:
        start = max(0, int(row["start_frame"]))
        end = min(n_frames, int(row["end_frame"]))
        if end > start:
            standard_scores[start:end] = float(row["score_prob"])
    for idx, row in enumerate(rows):
        start = min(n_frames, int(row["end_frame"]))
        next_end = n_frames if idx + 1 == len(rows) else min(n_frames, int(rows[idx + 1]["end_frame"]))
        if next_end > start:
            causal_scores[start:next_end] = float(row["score_prob"])
            causal_valid[start:next_end] = True

    np.save(output_dir / f"{video_id}_standard_frame_scores.npy", standard_scores)
    np.save(output_dir / f"{video_id}_causal_frame_scores.npy", causal_scores)
    np.save(output_dir / f"{video_id}_causal_valid_mask.npy", causal_valid)

    standard_auc, standard_ap = _auc_ap(standard_scores, gt)
    causal_auc, causal_ap = _auc_ap(causal_scores[causal_valid], gt[causal_valid]) if causal_valid.any() else (None, None)
    score_values = np.array([r["score_prob"] for r in rows], dtype=np.float32)

    if video_id == "Abuse028_x264":
        print("Abuse028 windows 2~6")
        for row in rows:
            if 2 <= int(row["window_index"]) <= 6:
                print(
                    f"window={row['window_index']} frames={row['start_frame']}-{row['end_frame']} "
                    f"time={row['start_sec']:.2f}-{row['end_sec']:.2f}s "
                    f"score_prob={row['score_prob']:.6f}"
                )

    video_duration_sec = n_frames / fps
    rtf_processing = processing_sec / video_duration_sec if video_duration_sec > 0 else math.inf
    rtf_full = (data_loading_sec + processing_sec) / video_duration_sec if video_duration_sec > 0 else math.inf
    avg_window_ms = 1000.0 * processing_sec / max(len(rows), 1)
    steady_window_ms = 1000.0 * steady_processing_sec / max(len(rows) - first_chunk_scored, 1)

    print(
        f"video_id={video_id} n_frames={n_frames} fps={fps} num_windows={len(rows)} "
        f"score_min={float(score_values.min()) if len(score_values) else math.nan:.6f} "
        f"score_max={float(score_values.max()) if len(score_values) else math.nan:.6f} "
        f"score_mean={float(score_values.mean()) if len(score_values) else math.nan:.6f} "
        f"GT_abnormal_frames={int(gt.sum())} "
        f"standard_auc={standard_auc if standard_auc is not None else 'N/A'} "
        f"standard_ap={standard_ap if standard_ap is not None else 'N/A'} "
        f"causal_auc={causal_auc if causal_auc is not None else 'N/A'} "
        f"causal_ap={causal_ap if causal_ap is not None else 'N/A'} "
        f"proc_sec={processing_sec:.2f} rtf_proc={rtf_processing:.3f} "
        f"rtf_full={rtf_full:.3f} avg_win_ms={avg_window_ms:.1f} "
        f"steady_win_ms={steady_window_ms:.1f}"
    )
    return {
        "video_id": video_id,
        "n_frames": n_frames,
        "fps": fps,
        "num_windows": len(rows),
        "standard_auc": standard_auc,
        "standard_ap": standard_ap,
        "causal_auc": causal_auc,
        "causal_ap": causal_ap,
        "processing_sec": processing_sec,
        "data_loading_sec": data_loading_sec,
        "rtf_processing": rtf_processing,
        "rtf_full": rtf_full,
        "avg_window_ms": avg_window_ms,
        "steady_window_ms": steady_window_ms,
        "gt": gt,
        "standard_scores": standard_scores,
        "causal_scores": causal_scores,
        "causal_valid": causal_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--feature-cache-root", default="")
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--device", default="cuda")
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
    )
    grouped = group_video_chunks(dataset.samples)
    if args.video_id:
        selected_video_id = Path(args.video_id).stem
        if selected_video_id not in grouped:
            raise ValueError(f"video_id={selected_video_id!r} not found in dataset")
        grouped = {selected_video_id: grouped[selected_video_id]}

    all_gt = []
    all_standard = []
    all_causal_gt = []
    all_causal = []
    videos = []
    failed_videos: List[dict] = []
    for video_id, refs in tqdm(grouped.items(), desc="Infer videos"):
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
            )
        except Exception as exc:  # keep one bad video from killing the whole run
            failed_videos.append({"video_id": video_id, "error": str(exc)[:300]})
            print(f"FAILED video={video_id}: {exc}")
            continue
        videos.append({k: v for k, v in result.items() if k not in {"gt", "standard_scores", "causal_scores", "causal_valid"}})
        all_gt.append(result["gt"])
        all_standard.append(result["standard_scores"])
        mask = result["causal_valid"]
        all_causal_gt.append(result["gt"][mask])
        all_causal.append(result["causal_scores"][mask])

    gt_all = np.concatenate(all_gt) if all_gt else np.empty(0, dtype=np.int64)
    standard_all = np.concatenate(all_standard) if all_standard else np.empty(0, dtype=np.float32)
    causal_gt_all = np.concatenate(all_causal_gt) if all_causal_gt else np.empty(0, dtype=np.int64)
    causal_all = np.concatenate(all_causal) if all_causal else np.empty(0, dtype=np.float32)
    global_standard_auc, global_standard_ap = _auc_ap(standard_all, gt_all)
    global_causal_auc, global_causal_ap = _auc_ap(causal_all, causal_gt_all)

    total_processing_sec = sum(float(v["processing_sec"]) for v in videos)
    total_data_sec = sum(float(v["data_loading_sec"]) for v in videos)
    total_video_sec = sum(float(v["n_frames"]) / float(v["fps"]) for v in videos)
    total_windows = sum(int(v["num_windows"]) for v in videos)

    metrics = {
        "num_videos": len(videos),
        "num_failed_videos": len(failed_videos),
        "failed_videos": failed_videos,
        "global_standard_auc": global_standard_auc,
        "global_standard_ap": global_standard_ap,
        "global_causal_auc": global_causal_auc,
        "global_causal_ap": global_causal_ap,
        "total_processing_sec": total_processing_sec,
        "total_data_loading_sec": total_data_sec,
        "total_video_sec": total_video_sec,
        "rtf_processing": total_processing_sec / total_video_sec if total_video_sec > 0 else None,
        "rtf_full": (total_processing_sec + total_data_sec) / total_video_sec if total_video_sec > 0 else None,
        "avg_window_ms": 1000.0 * total_processing_sec / max(total_windows, 1),
        "videos": videos,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "videos"}, indent=2))


if __name__ == "__main__":
    main()
