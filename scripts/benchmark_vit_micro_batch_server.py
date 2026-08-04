#!/usr/bin/env python
"""Benchmark ViT micro-batch sizes under FlashAttention-2.

Each micro-batch size runs in an independent subprocess to avoid
GPU memory cache contamination.

Server usage:
    python scripts/benchmark_vit_micro_batch_server.py \
        --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
        --video-path /data3/wgq/Stream-vad/sample_video.mp4
"""
import argparse
import subprocess
import sys
import time


BENCH_SCRIPT = """
import argparse, sys, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--video-path", required=True)
parser.add_argument("--vit-micro-batch", type=int, default=1)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--measure", type=int, default=5)
args = parser.parse_args()

# load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None, low_cpu_mem_usage=True,
).cuda().eval()

processor = Qwen2VLProcessor.from_pretrained(args.model_path)

# load a real video chunk
from decord import VideoReader, cpu
vr = VideoReader(args.video_path, ctx=cpu(0))
n_frames = min(len(vr), 20 * 32)
indices = list(range(0, n_frames, max(1, len(vr) // (20 * 32))))[: 20 * 32]
frames = vr.get_batch(indices).asnumpy()
frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

# split into max_windows=32 clips of 20 frames each
max_w = 32
F = 20
n_clips = min(max_w, frames.shape[0] // F)
all_clips = [frames[i*F:(i+1)*F] for i in range(n_clips)]

processed = processor.image_processor(images=None, videos=all_clips, return_tensors="pt")
pv = processed["pixel_values_videos"].cuda()
gthw = processed["video_grid_thw"].cuda()

# ViT forward
visual = model.visual

# import compression
from temporal import TemporalTokenReducer
reducer = TemporalTokenReducer()

def run_vit(micro_size):
    patch_counts = gthw.detach().cpu().prod(dim=1).tolist()
    pixel_clips = torch.split(pv, patch_counts, dim=0)
    total = gthw.shape[0]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for start in range(0, total, micro_size):
        end = min(start + micro_size, total)
        micro_pv = torch.cat(pixel_clips[start:end], dim=0)
        micro_gthw = gthw[start:end]

        patches = visual.patch_embed(micro_pv)
        rotary = visual.rot_pos_emb(micro_gthw)
        mask, seqlens = reducer(patches, micro_gthw)
        patches = patches[mask]
        rotary = rotary[mask]
        cu = torch.nn.functional.pad(seqlens.cumsum(0), (1,0), value=0).int()

        for blk in visual.blocks:
            patches = blk(patches, cu_seqlens=cu, rotary_pos_emb=rotary)

        _ = visual.merger(patches)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed

# warmup
for _ in range(args.warmup):
    run_vit(args.vit_micro_batch)

# measure
times = []
mem_alloc = []
mem_reserved = []
for _ in range(args.measure):
    torch.cuda.reset_peak_memory_stats()
    t = run_vit(args.vit_micro_batch)
    times.append(t)
    mem_alloc.append(torch.cuda.max_memory_allocated() / 1024**3)
    mem_reserved.append(torch.cuda.max_memory_reserved() / 1024**3)

print(f"MB={args.vit_micro_batch} "
      f"time={np.mean(times):.3f}s±{np.std(times):.3f}s "
      f"mem_alloc={np.mean(mem_alloc):.2f}GB "
      f"mem_reserved={np.mean(mem_reserved):.2f}GB")
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    args = parser.parse_args()

    micro_sizes = [1, 2, 4, 8]

    print("| vit_micro_batch | time (s) | mem_alloc (GB) | mem_reserved (GB) |")
    print("|----------------|----------|----------------|-------------------|")

    for mb in micro_sizes:
        try:
            result = subprocess.run(
                [sys.executable, "-c", BENCH_SCRIPT,
                 "--model-path", args.model_path,
                 "--video-path", args.video_path,
                 "--vit-micro-batch", str(mb)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"| {mb} | OOM/ERROR | - | - |")
                print(f"  stderr: {result.stderr[-200:]}")
            else:
                # parse output
                line = result.stdout.strip().split("\n")[-1]
                print(f"| {mb} | {line} |")
        except subprocess.TimeoutExpired:
            print(f"| {mb} | TIMEOUT | - | - |")

    print("\nDone. Run on server with a real video clip.")


if __name__ == "__main__":
    main()
