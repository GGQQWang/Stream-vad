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
from pathlib import Path

import numpy as np
import torch

# ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


def load_and_run(model_path: str, attn_impl: str, all_clips):
    """Load model with given backend, run vision tower, return tokens."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=None, low_cpu_mem_usage=True,
    ).cuda().eval()

    # fixed 448×448 for reproducible comparison
    processor = Qwen2VLProcessor.from_pretrained(
        model_path, min_pixels=200704, max_pixels=200704,
    )
    processed = processor.image_processor(
        images=None, videos=all_clips, return_tensors="pt",
    )
    pv = processed["pixel_values_videos"].cuda()
    gthw = processed["video_grid_thw"].cuda()

    # use project's ViTForwarder (same path as training)
    from temporal import TemporalTokenReducer
    from vit_forwarder import ViTForwarder

    visual = model.visual
    vit = ViTForwarder(visual, TemporalTokenReducer()).cuda().eval()

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
    ):
        tokens, counts = vit.forward_batch_micro(
            pv, gthw, micro_batch_size=1,
        )

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

    assert torch.cuda.is_available()
    torch.cuda.init()

    from decord import VideoReader, cpu
    vr = VideoReader(args.video_path, ctx=cpu(0))
    target_frames = 20 * 4
    if len(vr) < target_frames:
        raise RuntimeError(f"Video has {len(vr)} frames, need {target_frames}")
    indices = list(range(target_frames))
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
    F = 20
    all_clips = [frames[i*F:(i+1)*F] for i in range(4)]
    assert len(all_clips) == 4, f"Expected 4 clips, got {len(all_clips)}"

    print(f"=== SDPA vs FA2 comparison (4 clips) ===\n")

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

    # shape mismatch is a hard failure
    if a.shape != b.shape:
        sys.exit(f"FAIL: token shapes differ: {a.shape} vs {b.shape}")

    # non-finite output is a hard failure
    if not a.isfinite().all():
        sys.exit("FAIL: SDPA output non-finite")
    if not b.isfinite().all():
        sys.exit("FAIL: FA2 output non-finite")

    # max_abs is reported but not a failure criterion (single outliers
    #  can exceed 1e-2 in deep BF16 ViT without invalidating the model)
    print(f"max abs error:   {max_abs:.6e}")

    # primary agreement criteria
    if cos < 0.995:
        sys.exit(f"FAIL: cosine similarity too low: {cos:.6f}")
    if rel_l2 > 0.05:
        sys.exit(f"FAIL: relative L2 too large: {rel_l2:.6e}")

    print("\nSDPA vs FA2: PASS (within BF16 tolerance)")


if __name__ == "__main__":
    main()
