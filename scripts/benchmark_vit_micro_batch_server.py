#!/usr/bin/env python
"""Frozen-ViT micro-batch benchmark (FlashAttention-2).

Measures the visual tower forward only (frozen in training).
Each micro-batch size runs in an independent subprocess.

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

assert torch.cuda.is_available()
torch.cuda.init()

# load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None, low_cpu_mem_usage=True,
).cuda().eval()

# fixed 448×448 for reproducible benchmark
processor = Qwen2VLProcessor.from_pretrained(
    args.model_path,
    min_pixels=200704,
    max_pixels=200704,
)

# check FA2 backend
visual = model.visual
vis_cls = type(visual.blocks[0].attn).__name__ if hasattr(visual.blocks[0], "attn") else type(visual.blocks[0]).__name__
print(f"Vision attention class: {vis_cls}")

from decord import VideoReader, cpu
vr = VideoReader(args.video_path, ctx=cpu(0))
target_frames = 20 * 32
if len(vr) < target_frames:
    raise RuntimeError(f"Video has {len(vr)} frames, benchmark needs {target_frames}")
indices = list(range(target_frames))
frames = vr.get_batch(indices).asnumpy()
frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

F = 20
all_clips = [frames[i*F:(i+1)*F] for i in range(target_frames // F)]
assert len(all_clips) == 32, f"Expected 32 clips, got {len(all_clips)}"

processed = processor.image_processor(images=None, videos=all_clips, return_tensors="pt")
pv = processed["pixel_values_videos"].cuda()
gthw = processed["video_grid_thw"].cuda()
print(f"grid_thw sample: {gthw[:2].tolist()}")

# use project's ViTForwarder
from temporal import TemporalTokenReducer
from vit_forwarder import ViTForwarder

vit = ViTForwarder(visual, TemporalTokenReducer()).cuda().eval()

def run_vit(micro_size):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
    ):
        tokens, counts = vit.forward_batch_micro(
            pv, gthw, micro_batch_size=micro_size,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    assert torch.isfinite(tokens).all(), "ViT output non-finite"
    return elapsed, tokens.shape[0]

# warmup
for _ in range(args.warmup):
    run_vit(args.vit_micro_batch)

# measure
times = []
mem_alloc = []
mem_reserved = []
for _ in range(args.measure):
    torch.cuda.reset_peak_memory_stats()
    t, n_tok = run_vit(args.vit_micro_batch)
    times.append(t)
    mem_alloc.append(torch.cuda.max_memory_allocated() / 1024**3)
    mem_reserved.append(torch.cuda.max_memory_reserved() / 1024**3)

print(f"MB={args.vit_micro_batch} "
      f"tokens={n_tok} "
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

    print("| vit_micro_batch | time (s) | mem_alloc (GB) | mem_reserved (GB) | tokens |")
    print("|----------------|----------|----------------|-------------------|--------|")

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
