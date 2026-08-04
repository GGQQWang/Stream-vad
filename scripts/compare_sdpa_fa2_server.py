#!/usr/bin/env python
"""Numerical comparison: SDPA vs FlashAttention-2 on the same input.

Loads each backend sequentially (never both at once) and compares
vision-tower output on identical data.

Server usage:
    python scripts/compare_sdpa_fa2_server.py \
        --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
        --video-path /data3/wgq/Stream-vad/sample_video.mp4
"""
import argparse
import sys

import numpy as np
import torch

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


def load_and_run(model_path: str, attn_impl: str, all_clips):
    """Load model with given backend, run vision tower, return tokens."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=None, low_cpu_mem_usage=True,
    ).cuda().eval()

    processor = Qwen2VLProcessor.from_pretrained(model_path)
    processed = processor.image_processor(
        images=None, videos=all_clips, return_tensors="pt",
    )
    pv = processed["pixel_values_videos"].cuda()
    gthw = processed["video_grid_thw"].cuda()

    from temporal import TemporalTokenReducer
    reducer = TemporalTokenReducer()

    visual = model.visual
    patches = visual.patch_embed(pv)
    rotary = visual.rot_pos_emb(gthw)
    mask, seqlens = reducer(patches, gthw)
    patches = patches[mask]
    rotary = rotary[mask]
    cu = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0), value=0).int()

    for blk in visual.blocks:
        patches = blk(patches, cu_seqlens=cu, rotary_pos_emb=rotary)

    tokens = visual.merger(patches)

    # also check text attention
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    txt_cls = type(lm.layers[0].self_attn).__name__
    vis_cls = type(visual.blocks[0].attn).__name__ if hasattr(visual.blocks[0], "attn") else type(visual.blocks[0]).__name__

    result = {
        "tokens": tokens.detach().float().cpu(),
        "vis_attn_cls": vis_cls,
        "txt_attn_cls": txt_cls,
    }
    del model
    torch.cuda.empty_cache()
    return result


def cosine_sim(a, b):
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    return (a_f @ b_f).item() / (a_f.norm().item() * b_f.norm().item() + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # ---- prepare identical input ----
    from decord import VideoReader, cpu
    vr = VideoReader(args.video_path, ctx=cpu(0))
    n_frames = min(len(vr), 20 * 4)  # small: 4 clips
    indices = list(range(0, n_frames, max(1, len(vr) // (20 * 4))))[: 20 * 4]
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
    F = 20
    n_clips = min(4, frames.shape[0] // F)
    all_clips = [frames[i*F:(i+1)*F] for i in range(n_clips)]

    print(f"=== SDPA vs FA2 comparison ({n_clips} clips) ===\n")

    # ---- SDPA ----
    print("Loading SDPA model ...")
    sdpa = load_and_run(args.model_path, "sdpa", all_clips)
    print(f"  vis={sdpa['vis_attn_cls']}  txt={sdpa['txt_attn_cls']}")

    # ---- FA2 ----
    print("Loading FA2 model ...")
    fa2 = load_and_run(args.model_path, "flash_attention_2", all_clips)
    print(f"  vis={fa2['vis_attn_cls']}  txt={fa2['txt_attn_cls']}")

    # ---- compare ----
    a = sdpa["tokens"]
    b = fa2["tokens"]

    print(f"\nshape match:     {a.shape == b.shape}")
    print(f"SDPA finite:     {a.isfinite().all().item()}")
    print(f"FA2 finite:      {b.isfinite().all().item()}")
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    rel_l2 = (a - b).norm().item() / (a.norm().item() + 1e-12)
    cos = cosine_sim(a, b)
    print(f"max abs error:   {max_abs:.6e}")
    print(f"mean abs error:  {mean_abs:.6e}")
    print(f"relative L2:     {rel_l2:.6e}")
    print(f"cosine similarity: {cos:.6f}")

    # Thresholds for BF16 numerical agreement
    if max_abs > 1e-2:
        print("\nWARNING: max abs error > 1e-2 — unexpected for BF16")
        sys.exit(1)

    if cos < 0.999:
        print(f"\nWARNING: cosine similarity {cos:.6f} < 0.999")
        sys.exit(1)

    print("\nSDPA vs FA2: PASS (within BF16 tolerance)")


if __name__ == "__main__":
    main()
