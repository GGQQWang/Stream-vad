"""Group A: deterministic, raw Qwen2-VL video-window score generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frame_score_eval import auc_ap, frame_scores_from_windows
from hivau_dataset import HIVAUDataset
from infer_stage1_ucf import (
    FRAMES_PER_CLIP, MAX_PIXELS, MIN_PIXELS, SAMPLE_INTERVAL,
    load_gt, normalize_manifest, save_window_csv,
)
from mil_utils import group_video_chunks
from qwen_vl_compat import language_hidden_size, load_vl_model, load_vl_processor, model_family
from tools.ablation_5_core import ZERO_SHOT_PROMPT, parse_generated_score


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_window_score(
    model, processor, clip: torch.Tensor, device: torch.device,
) -> tuple[str, float | None, float, str | None]:
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": ZERO_SHOT_PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video = clip.permute(0, 2, 3, 1).contiguous().numpy()
    inputs = processor(text=[prompt], videos=[video], padding=True, return_tensors="pt").to(device)
    if "video_grid_thw" not in inputs or int(inputs["video_grid_thw"].shape[0]) != 1:
        raise RuntimeError("Qwen2-VL processor did not produce exactly one video window")
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1, use_cache=True)
    _sync(device)
    generation_ms = 1000.0 * (time.perf_counter() - start)
    continuation = output[0, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode([continuation], skip_special_tokens=True)[0]
    score, error = parse_generated_score(raw)
    return raw, score, generation_ms, error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    if model_family(config) != "qwen2_vl" or language_hidden_size(config) != 3584:
        raise ValueError("Group A requires Qwen2-VL-7B (language hidden size 3584)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = normalize_manifest(
        args.test_manifest, output_dir, args.video_id, official_gt_only=True,
    )
    dataset = HIVAUDataset(
        manifest_path, args.video_root, total_sampled_frames=FRAMES_PER_CLIP,
        sample_interval=SAMPLE_INTERVAL, max_windows=1,
    )
    grouped = group_video_chunks(dataset.samples)
    expected = set(json.loads(manifest_path.read_text()))
    missing = sorted(expected - set(grouped))
    if missing:
        raise RuntimeError(f"zero-shot dataset skipped {len(missing)} manifest videos: {missing[:20]}")

    device = torch.device(args.device)
    model = load_vl_model(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = load_vl_processor(args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    videos = []
    all_gt, all_standard, all_causal_gt, all_causal = [], [], [], []
    all_processing_ms, all_generation_ms = [], []
    generation_failures = 0
    failed_videos = []
    total_data_sec = 0.0
    with (output_dir / "window_generations.jsonl").open("w", encoding="utf-8") as generation_log:
        for video_id, refs in grouped.items():
            meta = dataset.samples[refs[0].index]
            n_frames, fps = int(meta["n_frames"]), float(meta["fps"])
            gt = load_gt(args.gt_root, video_id, n_frames)
            rows = []
            video_processing_ms = []
            video_generation_ms = []
            for ref in refs:
                load_start = time.perf_counter()
                sample = dataset[ref.index]
                total_data_sec += time.perf_counter() - load_start
                if int(sample["valid_mask"].sum()) != 1:
                    raise ValueError(f"{video_id}: zero-shot inference requires one valid window per sample")
                clip = sample["frames"][0]
                _sync(device)
                processing_start = time.perf_counter()
                raw, score, generation_ms, error = generate_window_score(model, processor, clip, device)
                _sync(device)
                processing_ms = 1000.0 * (time.perf_counter() - processing_start)
                row = {
                    "video_id": video_id,
                    "window_index": int(sample["chunk_start"]),
                    "start_frame": int(sample["window_start_frames"][0]),
                    "end_frame": int(sample["valid_end_frames"][0]),
                    "score_logit": "", "score_prob": score,
                }
                row["start_sec"] = row["start_frame"] / fps
                row["end_sec"] = row["end_frame"] / fps
                rows.append(row)
                record = {**row, "raw_text": raw, "parse_error": error,
                          "generation_time_ms": generation_ms,
                          "window_processing_ms": processing_ms}
                generation_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                generation_log.flush()
                video_processing_ms.append(processing_ms)
                video_generation_ms.append(generation_ms)
                if error:
                    generation_failures += 1
                    print(f"GENERATION_PARSE_FAILED video={video_id} window={row['window_index']} "
                          f"reason={error} raw={raw!r}")
            rows.sort(key=lambda item: item["window_index"])
            if not rows or rows[0]["start_frame"] != 0 or rows[-1]["end_frame"] != n_frames:
                raise RuntimeError(f"{video_id}: incomplete zero-shot window timeline")
            if any(prev["end_frame"] != cur["start_frame"] for prev, cur in zip(rows, rows[1:])):
                raise RuntimeError(f"{video_id}: non-contiguous zero-shot windows")
            save_window_csv(output_dir / f"{video_id}_window_scores.csv", rows)
            video_metrics = {"video_id": video_id, "n_frames": n_frames, "fps": fps,
                             "num_windows": len(rows),
                             "mean_window_processing_ms": float(np.mean(video_processing_ms)),
                             "p95_window_processing_ms": float(np.percentile(video_processing_ms, 95)),
                             "mean_generation_latency_ms": float(np.mean(video_generation_ms))}
            all_processing_ms.extend(video_processing_ms)
            all_generation_ms.extend(video_generation_ms)
            if any(row["score_prob"] is None for row in rows):
                failed_videos.append({"video_id": video_id, "error": "unparseable generated window score"})
                video_metrics.update({"standard_auc": None, "standard_ap": None,
                                      "causal_auc": None, "causal_ap": None})
            else:
                standard, causal, valid = frame_scores_from_windows(rows, n_frames)
                np.save(output_dir / f"{video_id}_standard_frame_scores.npy", standard)
                np.save(output_dir / f"{video_id}_causal_frame_scores.npy", causal)
                np.save(output_dir / f"{video_id}_causal_valid_mask.npy", valid)
                video_metrics["standard_auc"], video_metrics["standard_ap"] = auc_ap(standard, gt)
                video_metrics["causal_auc"], video_metrics["causal_ap"] = (
                    auc_ap(causal[valid], gt[valid]) if valid.any() else (None, None)
                )
                all_gt.append(gt)
                all_standard.append(standard)
                all_causal_gt.append(gt[valid])
                all_causal.append(causal[valid])
            videos.append(video_metrics)

    complete = not failed_videos and len(videos) == len(expected)
    gt_all = np.concatenate(all_gt) if all_gt else np.empty(0, dtype=np.int64)
    standard_all = np.concatenate(all_standard) if all_standard else np.empty(0, dtype=np.float32)
    causal_gt_all = np.concatenate(all_causal_gt) if all_causal_gt else np.empty(0, dtype=np.int64)
    causal_all = np.concatenate(all_causal) if all_causal else np.empty(0, dtype=np.float32)
    standard_auc, standard_ap = auc_ap(standard_all, gt_all) if complete else (None, None)
    causal_auc, causal_ap = auc_ap(causal_all, causal_gt_all) if complete else (None, None)
    total_video_sec = sum(video["n_frames"] / video["fps"] for video in videos)
    metrics = {
        "baseline": "Zero-shot Generative Baseline", "prompt": ZERO_SHOT_PROMPT,
        "num_videos": len(videos), "num_failed_videos": len(failed_videos),
        "failed_videos": failed_videos,
        "num_generation_failures": generation_failures,
        "generation_failure_rate": generation_failures / max(len(all_processing_ms), 1),
        "global_standard_auc": standard_auc, "global_standard_ap": standard_ap,
        "global_causal_auc": causal_auc, "global_causal_ap": causal_ap,
        "mean_window_processing_ms": float(np.mean(all_processing_ms)) if all_processing_ms else None,
        "p95_window_processing_ms": float(np.percentile(all_processing_ms, 95)) if all_processing_ms else None,
        "mean_generation_latency_ms": float(np.mean(all_generation_ms)) if all_generation_ms else None,
        "p95_generation_latency_ms": float(np.percentile(all_generation_ms, 95)) if all_generation_ms else None,
        "rtf_processing": sum(all_processing_ms) / 1000 / total_video_sec if total_video_sec else None,
        "total_data_loading_sec": total_data_sec, "inference_max_windows": 1,
        "videos": videos,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "videos"}, indent=2))
    if args.strict and not complete:
        raise RuntimeError("zero-shot inference has unparseable scores; no global benchmark metric reported")


if __name__ == "__main__":
    main()
