#!/usr/bin/env python
"""Diagnose numerical differences between Qwen2-VL attention backends.

Compares eager / sdpa / flash_attention_2 on identical input through
the vision tower, both with and without TemporalTokenReducer.

Server usage:
    python scripts/diagnose_attention_backends_server.py \
        --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
        --video-path /data3/wgq/Stream-vad/sample_video.mp4 \
        --output-dir ./fa2_diag

NOT YET RUN ON SERVER — this script has only passed local static checks.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BACKENDS = ["eager", "sdpa", "flash_attention_2"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tensor_checksum(t: torch.Tensor) -> str:
    """Stable hash of a CPU tensor's raw bytes."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _layer_indices(model) -> List[int]:
    """Pick representative layer indices (first, ~1/4, mid, ~3/4, last)."""
    n = len(model.visual.blocks)
    return [0, n // 4, n // 2, 3 * n // 4, n - 1]


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute comparison metrics.  Cosine uses float64 accumulation."""
    if a.dtype == torch.bool or a.dtype in (torch.int32, torch.int64):
        exact = torch.equal(a.cpu(), b.cpu())
        return {"shape_match": a.shape == b.shape, "exact_match": exact}

    # float64 for stable cosine on large tensors
    x = a.detach().reshape(-1).cpu().to(torch.float64)
    y = b.detach().reshape(-1).cpu().to(torch.float64)
    diff = x - y

    norm_x = torch.linalg.vector_norm(x)
    denom = (norm_x * torch.linalg.vector_norm(y)).clamp_min(1e-30)
    cosine = float((torch.dot(x, y) / denom).clamp(-1.0, 1.0))

    return {
        "shape_match": a.shape == b.shape,
        "finite": a.isfinite().all().item() and b.isfinite().all().item(),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": float(torch.linalg.vector_norm(diff) / norm_x.clamp_min(1e-30)),
        "cosine": cosine,
    }


# ---------------------------------------------------------------------------
# single-backend runner
# ---------------------------------------------------------------------------

def _run_official(
    visual: torch.nn.Module,
    pv: torch.Tensor,
    gthw: torch.Tensor,
    save_layers: Optional[List[int]],
) -> Dict[str, torch.Tensor]:
    """Official Qwen2-VL vision forward — no temporal compression."""
    results = {}

    patches = visual.patch_embed(pv)                              # [L, 1280]
    results["patch_embed"] = patches.detach()

    rotary = visual.rot_pos_emb(gthw)                             # [L, 1280]
    results["rotary_pos_emb"] = rotary.detach()

    tg = gthw[:, 0]
    hg = gthw[:, 1]
    wg = gthw[:, 2]
    seqlens = torch.repeat_interleave(hg * wg, tg)                # [sum(t_i)]
    seqlens = seqlens.to(device=gthw.device, dtype=torch.int32)
    results["seqlens"] = seqlens.detach().cpu()

    cu = F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
    results["cu_seqlens"] = cu.detach().cpu()

    keep = save_layers or []
    for i, blk in enumerate(visual.blocks):
        patches = blk(patches, cu_seqlens=cu, rotary_pos_emb=rotary)
        if i in keep:
            results[f"block_{i}"] = patches.detach()

    tokens = visual.merger(patches)
    results["merger"] = tokens.detach()
    return {k: v.float().cpu() if v.dtype != torch.bool and v.dtype not in (torch.int32, torch.int64)
            else v.cpu() for k, v in results.items()}


def _run_compressed(
    visual: torch.nn.Module,
    pv: torch.Tensor,
    gthw: torch.Tensor,
    save_layers: Optional[List[int]],
) -> Dict[str, torch.Tensor]:
    """Stream-vad ViTForwarder path — with TemporalTokenReducer."""
    from temporal import TemporalTokenReducer

    reducer = TemporalTokenReducer()
    results = {}

    patches = visual.patch_embed(pv)
    results["patch_embed"] = patches.detach()

    rotary = visual.rot_pos_emb(gthw)
    results["rotary_pos_emb"] = rotary.detach()

    # temporal reduction
    mask, seqlens_comp = reducer(patches, gthw)
    results["mask"] = mask.detach().cpu()
    results["seqlens_comp"] = seqlens_comp.detach().cpu()

    patches = patches[mask]
    rotary = rotary[mask]

    cu = F.pad(seqlens_comp.cumsum(dim=0), (1, 0), value=0).int()
    results["cu_seqlens"] = cu.detach().cpu()

    keep = save_layers or []
    for i, blk in enumerate(visual.blocks):
        patches = blk(patches, cu_seqlens=cu, rotary_pos_emb=rotary)
        if i in keep:
            results[f"block_{i}"] = patches.detach()

    tokens = visual.merger(patches)
    results["merger"] = tokens.detach()
    return {k: v.float().cpu() if v.dtype != torch.bool and v.dtype not in (torch.int32, torch.int64)
            else v.cpu() for k, v in results.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=20)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--mode", choices=["official", "compressed", "both"], default="both")
    parser.add_argument("--vit-micro-batch", type=int, default=1)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-intermediates", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CUDA init before decord ----
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.init()

    # ---- prepare input once ----
    from decord import VideoReader, cpu as decord_cpu

    target_frames = args.num_clips * args.frames_per_clip
    vr = VideoReader(args.video_path, ctx=decord_cpu(0))
    if len(vr) < target_frames:
        sys.exit(f"Video has {len(vr)} frames, need {target_frames}")
    indices = list(range(target_frames))
    frames = vr.get_batch(indices).asnumpy()
    del vr

    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)
    all_clips = [
        frames_t[i * args.frames_per_clip : (i + 1) * args.frames_per_clip]
        for i in range(args.num_clips)
    ]

    # processor once
    from transformers import Qwen2VLProcessor
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processed = processor.image_processor(
        images=None, videos=all_clips, return_tensors="pt",
    )
    pv_cpu = processed["pixel_values_videos"]
    gthw_cpu = processed["video_grid_thw"]

    print(f"\n=== Input ===")
    print(f"  pixel_values: {pv_cpu.shape}  dtype={pv_cpu.dtype}  checksum={_tensor_checksum(pv_cpu)}")
    print(f"  grid_thw:     {gthw_cpu.shape}  {gthw_cpu.tolist()}")
    print(f"  checksum:     {_tensor_checksum(gthw_cpu)}")
    print(f"  num_clips={args.num_clips}  frames_per_clip={args.frames_per_clip}")
    print(f"  min_pixels={args.min_pixels}  max_pixels={args.max_pixels}")

    input_meta = {
        "num_clips": args.num_clips,
        "frames_per_clip": args.frames_per_clip,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "pv_shape": list(pv_cpu.shape),
        "pv_dtype": str(pv_cpu.dtype),
        "gthw": gthw_cpu.tolist(),
    }
    if out_dir:
        json.dump(input_meta, (out_dir / "input_metadata.json").open("w"), indent=2)

    # ---- run each backend ----
    from transformers import Qwen2VLForConditionalGeneration

    layer_idx = _layer_indices(
        Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=None, low_cpu_mem_usage=True,
        )
    )  # quick load to count layers, then delete
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nRepresentative layers: {layer_idx}")

    all_results: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    attn_classes: Dict[str, Tuple[str, str]] = {}  # backend → (vis_cls, txt_cls)

    for backend in BACKENDS:
        print(f"\n=== Loading {backend} ===")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=backend,
            device_map=None,
            low_cpu_mem_usage=True,
        ).cuda().eval()

        vis = model.visual
        vis_cls = type(vis.blocks[0].attn).__name__ if hasattr(vis.blocks[0], "attn") else type(vis.blocks[0]).__name__
        lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
        txt_cls = type(lm.layers[0].self_attn).__name__
        print(f"  vision attn: {vis_cls}")
        print(f"  text attn:   {txt_cls}")
        attn_classes[backend] = (vis_cls, txt_cls)

        # verify no silent fallback
        if backend == "flash_attention_2" and "FlashAttention2" not in vis_cls:
            sys.exit(f"FAIL: requested flash_attention_2 but got {vis_cls}")
        if backend == "sdpa" and "Sdpa" not in vis_cls:
            print(f"  WARNING: requested sdpa but got {vis_cls}")

        pv = pv_cpu.clone().cuda()
        gthw = gthw_cpu.clone().cuda()
        assert torch.equal(pv.cpu(), pv_cpu), f"pv mismatch for {backend}"
        assert torch.equal(gthw.cpu(), gthw_cpu), f"gthw mismatch for {backend}"

        results: Dict[str, Dict[str, torch.Tensor]] = {}

        # official path
        if args.mode in ("official", "both"):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
            ):
                results["official"] = _run_official(vis, pv, gthw, layer_idx)
            print(f"  official: merger shape={results['official']['merger'].shape}")

        # compressed path
        if args.mode in ("compressed", "both"):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
            ):
                results["compressed"] = _run_compressed(vis, pv, gthw, layer_idx)
            print(f"  compressed: merger shape={results['compressed']['merger'].shape}")

            # cross-check against project ViTForwarder
            from vit_forwarder import ViTForwarder
            vit = ViTForwarder(vis, None).cuda().eval()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
            ):
                ref_tokens, _ = vit.forward_batch_micro(pv, gthw)
            diag_tokens = results["compressed"]["merger"].to(ref_tokens.device).to(ref_tokens.dtype)
            max_err = (ref_tokens.float() - diag_tokens.float()).abs().max().item()
            print(f"  vs ViTForwarder: max_abs_error={max_err:.6e}")
            if max_err > 1e-4:
                print("  WARNING: diagnostic compressed output ≠ ViTForwarder output")

        all_results[backend] = results

        # save intermediates if requested
        if out_dir and args.save_intermediates:
            for path_name, tensors in results.items():
                for k, v in tensors.items():
                    torch.save(v, out_dir / f"{backend}_{path_name}_{k}.pt")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- compare ----
    print(f"\n{'='*100}")
    print("=== COMPARISON ===")
    print(f"{'='*100}")

    pairs = [
        ("eager vs sdpa", "eager", "sdpa"),
        ("eager vs FA2", "eager", "flash_attention_2"),
        ("sdpa vs FA2", "sdpa", "flash_attention_2"),
    ]

    all_metrics: List[dict] = []

    for pair_name, ba, bb in pairs:
        if ba not in all_results or bb not in all_results:
            continue
        print(f"\n--- {pair_name} ---")

        for path_name in all_results[ba].keys():
            tensors_a = all_results[ba][path_name]
            tensors_b = all_results[bb][path_name]

            keys = sorted(set(tensors_a.keys()) & set(tensors_b.keys()))
            for key in keys:
                a = tensors_a[key]
                b = tensors_b[key]
                m = _metrics(a, b)
                m["pair"] = pair_name
                m["path"] = path_name
                m["layer"] = key
                all_metrics.append(m)

                status = "✓" if m.get("shape_match", True) and m.get("finite", True) else "✗"
                if m.get("cosine") is not None:
                    print(f"  {status} {path_name:12s} {key:16s}  "
                          f"max_abs={m['max_abs']:.4e}  cos={m['cosine']:.6f}  "
                          f"rel_l2={m['relative_l2']:.4e}")
                else:
                    match = "match" if m.get("exact_match", False) else "DIFF"
                    print(f"  {status} {path_name:12s} {key:16s}  {match}")

    # ---- summary ----
    print(f"\n--- Summary ---")
    # find first layer with significant divergence
    first_bad = None
    for m in all_metrics:
        cos = m.get("cosine")
        if cos is not None and cos < 0.999:
            first_bad = m
            break
    if first_bad:
        print(f"First significant divergence: {first_bad['pair']} "
              f"{first_bad['path']}/{first_bad['layer']}  cos={first_bad['cosine']:.6f}")

    # divergence trend
    for pair_name, _, _ in pairs:
        depths = []
        for m in all_metrics:
            if m["pair"] == pair_name and m["layer"].startswith("block_"):
                depths.append((int(m["layer"].split("_")[1]), m.get("cosine", 0)))
        if depths:
            depths.sort()
            print(f"\n{pair_name} cosine by depth:")
            for d, c in depths:
                bar = "█" * int(max(0, (c if c > 0 else 0) * 40))
                print(f"  block_{d:2d}  cos={c:.6f}  {bar}")

    # save metrics
    if out_dir:
        json.dump(all_metrics, (out_dir / "metrics.json").open("w"), indent=2)

        import csv
        fieldnames = ["pair", "path", "layer", "shape_match", "finite",
                      "max_abs", "mean_abs", "relative_l2", "cosine"]
        with (out_dir / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_metrics)

        with (out_dir / "summary.txt").open("w") as f:
            f.write("Attention backend comparison summary\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Backends: {BACKENDS}\n")
            f.write(f"Mode: {args.mode}\n\n")
            for m in all_metrics:
                f.write(f"{m['pair']:20s} {m['path']:12s} {m['layer']:16s}  "
                        f"cos={m.get('cosine', 'N/A')}\n")

    # structural failures → non-zero exit
    for m in all_metrics:
        if not m.get("shape_match", True):
            sys.exit(f"FAIL: shape mismatch in {m['pair']} {m['path']}/{m['layer']}")
        if m.get("finite") is False:
            sys.exit(f"FAIL: non-finite in {m['pair']} {m['path']}/{m['layer']}")
    # mask/seqlens must match exactly across backends
    for pair_name, ba, bb in pairs:
        for path_name in all_results.get(ba, {}):
            for key in ["mask", "seqlens_comp", "cu_seqlens", "seqlens"]:
                if key in all_results[ba].get(path_name, {}) and key in all_results[bb].get(path_name, {}):
                    a = all_results[ba][path_name][key]
                    b = all_results[bb][path_name][key]
                    if not torch.equal(a, b):
                        sys.exit(f"FAIL: {key} differs between {ba} and {bb}")

    print("\n=== DONE (no structural failures) ===")


if __name__ == "__main__":
    main()
