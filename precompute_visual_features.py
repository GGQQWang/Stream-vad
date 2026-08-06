"""Precompute frozen Qwen2-VL visual features for Stream-vad training."""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, set_seed

from feature_cache import (
    build_feature_cache_metadata,
    load_feature_cache,
    save_feature_cache_atomic,
)
from hivau_dataset import HIVAUDataset, hivau_collate
from mil_utils import group_video_chunks
from pipeline_stage1 import StreamingVADGenerationModel, _verify_attention_backend


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


@torch.no_grad()
def _extract_chunk_features(
    model: StreamingVADGenerationModel,
    processor: Qwen2VLProcessor,
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frames = batch["frames"][0]
    valid_mask_cpu = batch["valid_mask"]
    valid_mask = valid_mask_cpu.to(device)
    valid_w = valid_mask_cpu[0].nonzero(as_tuple=True)[0]
    clips = [frames[int(w)] for w in valid_w.tolist()]
    if not clips:
        return torch.empty(0, model.llm_hidden)

    processed = processor.image_processor(
        images=None,
        videos=clips,
        return_tensors="pt",
    )
    pixel_values = processed["pixel_values_videos"].to(device)
    grid_thw = processed["video_grid_thw"].to(device)

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
        window_batch = model.extract_window_features(
            pixel_values, grid_thw, valid_mask, return_stats=False,
        )
    return window_batch[0, valid_w.to(device)].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--sample-interval", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--vit-micro-batch", type=int, default=1)
    parser.add_argument("--d-ssm", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--attn-implementation", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    compute_dtype = _dtype_from_name(args.dtype)
    save_dtype = compute_dtype
    os.makedirs(args.cache_root, exist_ok=True)

    print("Loading Qwen2-VL for frozen visual precompute ...")
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False
    _verify_attention_backend(qwen, args.attn_implementation)

    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = StreamingVADGenerationModel(
        qwen,
        d_ssm=args.d_ssm,
        llm_hidden=qwen.config.hidden_size,
        vit_micro_batch=args.vit_micro_batch,
    ).to(device)
    model.eval()

    dataset = HIVAUDataset(
        args.annotation_json,
        args.video_root,
        total_sampled_frames=args.frames_per_clip,
        sample_interval=args.sample_interval,
        max_windows=args.max_windows,
    )
    grouped = group_video_chunks(dataset.samples)

    n_done = 0
    n_skip = 0
    for video_id, refs in tqdm(grouped.items(), desc="Precompute videos"):
        first = dataset.samples[refs[0].index]
        metadata = build_feature_cache_metadata(
            video_id=video_id,
            n_windows=first["n_total_windows"],
            n_frames=first["n_frames"],
            fps=first["fps"],
            frames_per_clip=args.frames_per_clip,
            sample_interval=args.sample_interval,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            model_id=args.model_path,
        )
        try:
            load_feature_cache(
                args.cache_root,
                video_id=video_id,
                n_windows=first["n_total_windows"],
                n_frames=first["n_frames"],
                fps=first["fps"],
                frames_per_clip=args.frames_per_clip,
                sample_interval=args.sample_interval,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                model_id=args.model_path,
                map_location="cpu",
            )
            n_skip += 1
            continue
        except FileNotFoundError:
            pass

        parts: List[torch.Tensor] = []
        for ref in refs:
            batch = hivau_collate([dataset[ref.index]])
            parts.append(_extract_chunk_features(model, processor, batch, device, compute_dtype))

        compressed = torch.cat(parts, dim=0).to(save_dtype)
        if compressed.shape[0] != first["n_total_windows"]:
            raise RuntimeError(
                f"{video_id}: extracted {compressed.shape[0]} windows, "
                f"expected {first['n_total_windows']}"
            )
        save_feature_cache_atomic(
            args.cache_root,
            video_id=video_id,
            compressed_features=compressed,
            metadata=metadata,
        )
        n_done += 1

    print(f"Precompute done: wrote={n_done}, skipped={n_skip}, cache_root={args.cache_root}")


if __name__ == "__main__":
    main()
