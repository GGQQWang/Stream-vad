"""Benchmark LVLM inference strategies for Stage-1 UCF inference.

This script intentionally reuses the Stage-1 inference components from
``infer_stage1_ucf.py``.  It does not retrain, change model structure, or feed
generated text back into the anomaly score path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hivau_dataset import HIVAUDataset, hivau_collate
from infer_stage1_ucf import (  # noqa: E402
    FRAMES_PER_CLIP,
    MAX_PIXELS,
    MAX_WINDOWS,
    MIN_PIXELS,
    SAMPLE_INTERVAL,
    _auc_ap,
    _find_embed,
    load_gt,
    load_stage1_model,
    normalize_manifest,
)
from mil_utils import group_video_chunks  # noqa: E402


INFERENCE_MODES = {
    "forward": None,
    "ar16": 16,
    "ar64": 64,
}
LATENCY_SCOPE = "cached_visual_feature_to_prediction"


def max_new_tokens_for_mode(mode: str) -> int | None:
    if mode not in INFERENCE_MODES:
        raise ValueError(f"unknown inference mode: {mode}")
    return INFERENCE_MODES[mode]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_latency(records: Iterable[dict]) -> dict:
    measured = [r for r in records if not r.get("is_warmup", False)]
    latencies = [float(r["latency_ms"]) for r in measured]
    temporal = [float(r.get("temporal_time_ms", 0.0)) for r in measured]
    forward = [float(r.get("lvlm_forward_time_ms", 0.0)) for r in measured]
    generation = [float(r.get("generation_time_ms", 0.0)) for r in measured]
    total_sec = sum(latencies) / 1000.0
    total_window_video_sec = sum(float(r.get("window_duration_sec", 0.0)) for r in measured)
    return {
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "throughput_windows_per_sec": (len(latencies) / total_sec) if total_sec > 0 else None,
        "RTF": (total_sec / total_window_video_sec) if total_window_video_sec > 0 else None,
        "temporal_time_ms": float(np.mean(temporal)) if temporal else None,
        "lvlm_forward_time_ms": float(np.mean(forward)) if forward else None,
        "generation_time_ms": float(np.mean(generation)) if generation else None,
        "measured_windows": len(latencies),
        "measured_processing_sec": total_sec,
        "measured_video_sec": total_window_video_sec,
    }


def peak_memory_gb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    )


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def build_generation_inputs(model, embed_fn, tokenizer, states: torch.Tensor, prompt_text: str) -> dict:
    """Build the same state/prompt/query prefix used by forward_score_token."""
    llm_weight = embed_fn.weight
    llm_device = llm_weight.device
    llm_dtype = llm_weight.dtype
    states = states.to(device=llm_device, dtype=llm_dtype)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=llm_device)
    prompt_emb = embed_fn(prompt_ids_t).unsqueeze(0).expand(states.shape[0], -1, -1)
    query = model.score_query.to(device=llm_device, dtype=llm_dtype).reshape(1, 1, -1).expand(states.shape[0], 1, -1)
    inputs_embeds = torch.cat([states.unsqueeze(1), prompt_emb, query], dim=1)
    attention_mask = torch.ones(states.shape[0], inputs_embeds.shape[1], dtype=torch.bool, device=llm_device)
    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def decode_generated(tokenizer, sequences, max_new_tokens: int) -> List[str]:
    if sequences is None:
        return []
    if hasattr(sequences, "detach"):
        sequences = sequences.detach().cpu()
    if getattr(sequences, "ndim", 0) == 1:
        sequences = sequences.unsqueeze(0)
    # With inputs_embeds, transformers usually returns only generated ids.  If
    # a backend prepends prompt ids, keep only the continuation budget.
    if sequences.shape[-1] > max_new_tokens:
        sequences = sequences[:, -max_new_tokens:]
    if hasattr(tokenizer, "batch_decode"):
        return list(tokenizer.batch_decode(sequences, skip_special_tokens=True))
    return [tokenizer.decode(row.tolist()) for row in sequences]


def maybe_generate_texts(model, embed_fn, tokenizer, states, prompt_text: str, max_new_tokens: int | None) -> List[str]:
    if max_new_tokens is None:
        return []
    gen_inputs = build_generation_inputs(model, embed_fn, tokenizer, states, prompt_text)
    sequences = model.qwen.generate(
        **gen_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    return decode_generated(tokenizer, sequences, max_new_tokens)


def frame_scores_from_rows(rows: List[dict], n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return standard_scores, causal_scores, causal_valid


def run_video_benchmark(
    *,
    model,
    processor,
    tokenizer,
    dataset: HIVAUDataset,
    refs,
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    gt_root: str | Path,
    inference_mode: str,
    latency_f,
    text_f,
    warmup_state: dict,
) -> dict:
    video_id = refs[0].video_id
    first_meta = dataset.samples[refs[0].index]
    n_frames = int(first_meta["n_frames"])
    fps = float(first_meta["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{video_id}: invalid fps for RTF/window timing: {fps!r}")
    gt = load_gt(gt_root, video_id, n_frames)
    embed_fn = _find_embed(model.qwen)
    ssm_cache: dict = {}
    rows: List[dict] = []
    max_new_tokens = max_new_tokens_for_mode(inference_mode)

    with torch.no_grad():
        for chunk_i, ref in enumerate(refs):
            batch = hivau_collate([dataset[ref.index]])
            if "features" not in batch:
                raise RuntimeError(
                    "LVLM inference ablation expects cached visual features; "
                    "feature precompute/decode time is outside the latency scope."
                )
            valid_mask = batch["valid_mask"].to(device)
            valid_b, valid_w = valid_mask.nonzero(as_tuple=True)
            if len(valid_b) == 0:
                continue

            if (
                not warmup_state["memory_reset_done"]
                and warmup_state["seen_windows"] >= warmup_state["warmup_windows"]
            ):
                synchronize(device)
                reset_peak_memory(device)
                warmup_state["memory_reset_done"] = True
            chunk_is_warmup = not warmup_state["memory_reset_done"]

            synchronize(device)
            chunk_start = time.perf_counter()
            temporal_start = time.perf_counter()
            window_batch = batch["features"].to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                state_emb, _, _, ssm_cache = model.encode_window_features(
                    window_batch,
                    valid_mask,
                    batch["video_id"],
                    ssm_cache,
                    training=False,
                )
            synchronize(device)
            temporal_ms_total = (time.perf_counter() - temporal_start) * 1000.0

            states = state_emb[valid_b, valid_w]
            forward_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                logits = model.forward_score_token(
                    states,
                    embed_fn,
                    tokenizer,
                    prompt_text=prompt_text,
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{video_id}: non-finite score logits in chunk {chunk_i}")
            synchronize(device)
            forward_ms_total = (time.perf_counter() - forward_start) * 1000.0

            generation_ms_total = 0.0
            generated_texts: List[str] = []
            if max_new_tokens is not None:
                generation_start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                    generated_texts = maybe_generate_texts(
                        model,
                        embed_fn,
                        tokenizer,
                        states,
                        prompt_text,
                        max_new_tokens,
                    )
                synchronize(device)
                generation_ms_total = (time.perf_counter() - generation_start) * 1000.0

            synchronize(device)
            chunk_latency_ms = (time.perf_counter() - chunk_start) * 1000.0

            probs = torch.sigmoid(logits).detach().float().cpu()
            logits_cpu = logits.detach().float().cpu()
            start_frames = batch["window_start_frames"]
            end_frames = batch["valid_end_frames"]
            valid_count = max(int(len(valid_b)), 1)
            for i, (b_t, w_t) in enumerate(zip(valid_b.cpu(), valid_w.cpu())):
                b = int(b_t.item())
                w = int(w_t.item())
                start_frame = int(start_frames[b, w].item())
                end_frame = int(end_frames[b, w].item())
                window_idx = int(batch["chunk_start"][b]) + w
                score = float(probs[i].item())
                gt_label = int(gt[start_frame:min(end_frame, n_frames)].max()) if end_frame > start_frame else 0
                is_warmup = chunk_is_warmup
                row = {
                    "video_id": video_id,
                    "window_idx": window_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "score_logit": float(logits_cpu[i].item()),
                    "score_prob": score,
                    "score": score,
                    "gt_label": gt_label,
                }
                rows.append(row)
                latency_record = {
                    "video_id": video_id,
                    "window_idx": window_idx,
                    "latency_ms": chunk_latency_ms / valid_count,
                    "temporal_time_ms": temporal_ms_total / valid_count,
                    "lvlm_forward_time_ms": forward_ms_total / valid_count,
                    "generation_time_ms": generation_ms_total / valid_count,
                    "is_warmup": is_warmup,
                    "latency_scope": LATENCY_SCOPE,
                    "window_duration_sec": max(0, end_frame - start_frame) / fps,
                }
                latency_f.write(json.dumps(latency_record) + "\n")
                if max_new_tokens is not None:
                    text_f.write(json.dumps({
                        "video_id": video_id,
                        "window_idx": window_idx,
                        "score": score,
                        "gt_label": gt_label,
                        "generated_text": generated_texts[i] if i < len(generated_texts) else "",
                        "max_new_tokens": max_new_tokens,
                    }, ensure_ascii=False) + "\n")
                warmup_state["seen_windows"] += 1

    ssm_cache.pop(video_id, None)
    rows.sort(key=lambda r: int(r["window_idx"]))
    standard_scores, causal_scores, causal_valid = frame_scores_from_rows(rows, n_frames)
    standard_auc, standard_ap = _auc_ap(standard_scores, gt)
    causal_auc, causal_ap = _auc_ap(causal_scores[causal_valid], gt[causal_valid]) if causal_valid.any() else (None, None)
    return {
        "video_id": video_id,
        "n_frames": n_frames,
        "fps": fps,
        "num_windows": len(rows),
        "standard_auc": standard_auc,
        "standard_ap": standard_ap,
        "causal_auc": causal_auc,
        "causal_ap": causal_ap,
        "gt": gt,
        "standard_scores": standard_scores,
        "causal_scores": causal_scores,
        "causal_valid": causal_valid,
    }


def read_jsonl(path: Path) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", default="")
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", default="/data3/wgq/outputs/lvlm_inference_ablation")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-mode", required=True, choices=sorted(INFERENCE_MODES))
    parser.add_argument("--warmup-windows", type=int, default=30)
    args = parser.parse_args()

    args.device = torch.device(args.device)
    run_dir = Path(args.output_dir) / args.inference_mode
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized_manifest = normalize_manifest(args.test_manifest, run_dir, args.video_id)
    model, processor, tokenizer, dtype, prompt_text = load_stage1_model(args)
    model.eval()

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
    )
    grouped = group_video_chunks(dataset.samples)
    if args.video_id:
        selected_video_id = Path(args.video_id).stem
        if selected_video_id not in grouped:
            raise ValueError(f"video_id={selected_video_id!r} not found in dataset")
        grouped = {selected_video_id: grouped[selected_video_id]}

    warmup_state = {
        "warmup_windows": max(0, int(args.warmup_windows)),
        "seen_windows": 0,
        "memory_reset_done": False,
    }
    if warmup_state["warmup_windows"] == 0:
        reset_peak_memory(args.device)
        warmup_state["memory_reset_done"] = True

    all_gt: List[np.ndarray] = []
    all_standard: List[np.ndarray] = []
    all_causal_gt: List[np.ndarray] = []
    all_causal: List[np.ndarray] = []
    videos: List[dict[str, Any]] = []
    failed_videos: List[dict[str, str]] = []
    latency_path = run_dir / "latency_per_window.jsonl"
    text_path = run_dir / "text_outputs.jsonl"

    wall_start = time.perf_counter()
    with open(latency_path, "w") as latency_f:
        text_f_cm = open(text_path, "w") if max_new_tokens_for_mode(args.inference_mode) is not None else None
        try:
            for video_id, refs in tqdm(grouped.items(), desc=f"Benchmark {args.inference_mode}"):
                try:
                    result = run_video_benchmark(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        refs=refs,
                        device=args.device,
                        dtype=dtype,
                        prompt_text=prompt_text,
                        gt_root=args.gt_root,
                        inference_mode=args.inference_mode,
                        latency_f=latency_f,
                        text_f=text_f_cm,
                        warmup_state=warmup_state,
                    )
                except Exception as exc:
                    failed_videos.append({"video_id": video_id, "error": str(exc)[:300]})
                    print(f"FAILED video={video_id}: {exc}")
                    continue
                videos.append({k: v for k, v in result.items() if k not in {"gt", "standard_scores", "causal_scores", "causal_valid"}})
                all_gt.append(result["gt"])
                all_standard.append(result["standard_scores"])
                mask = result["causal_valid"]
                all_causal_gt.append(result["gt"][mask])
                all_causal.append(result["causal_scores"][mask])
        finally:
            if text_f_cm is not None:
                text_f_cm.close()
    synchronize(args.device)
    total_wall_time_sec = time.perf_counter() - wall_start
    if not warmup_state["memory_reset_done"]:
        # The dataset was shorter than warmup_windows, so no formal latency or
        # peak-memory region exists.
        reset_peak_memory(args.device)
        warmup_state["memory_reset_done"] = True

    gt_all = np.concatenate(all_gt) if all_gt else np.empty(0, dtype=np.int64)
    standard_all = np.concatenate(all_standard) if all_standard else np.empty(0, dtype=np.float32)
    causal_gt_all = np.concatenate(all_causal_gt) if all_causal_gt else np.empty(0, dtype=np.int64)
    causal_all = np.concatenate(all_causal) if all_causal else np.empty(0, dtype=np.float32)
    global_standard_auc, global_standard_ap = _auc_ap(standard_all, gt_all)
    global_causal_auc, global_causal_ap = _auc_ap(causal_all, causal_gt_all)
    latency_records = read_jsonl(latency_path)
    latency_metrics = aggregate_latency(latency_records)
    effective_warmup_windows = sum(1 for r in latency_records if r.get("is_warmup", False))
    peak_allocated, peak_reserved = peak_memory_gb(args.device)

    rtf_note = None
    if latency_metrics["RTF"] is None:
        rtf_note = "RTF unavailable because no non-warmup window duration could be computed from manifest fps/n_frames."

    metrics = {
        "inference_mode": args.inference_mode,
        "max_new_tokens": max_new_tokens_for_mode(args.inference_mode),
        "latency_scope": LATENCY_SCOPE,
        "warmup_windows_requested": warmup_state["warmup_windows"],
        "warmup_windows_effective": effective_warmup_windows,
        "num_videos": len(videos),
        "num_failed_videos": len(failed_videos),
        "failed_videos": failed_videos,
        "global_standard_auc": global_standard_auc,
        "global_standard_ap": global_standard_ap,
        "global_causal_auc": global_causal_auc,
        "global_causal_ap": global_causal_ap,
        "peak_memory_allocated_gb": peak_allocated,
        "peak_memory_reserved_gb": peak_reserved,
        "total_wall_time_sec": total_wall_time_sec,
        "rtf_note": rtf_note,
        "videos": videos,
        **latency_metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "videos"}, indent=2))


if __name__ == "__main__":
    main()
