#!/usr/bin/env python
"""Verify FlashAttention-2 backend is active in a loaded Qwen2-VL model.

Server usage:
    python scripts/test_fa2_backend_server.py \
        --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct
"""
import argparse
import sys

import torch
from transformers import Qwen2VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    print("=== Flash-Attention-2 Backend Test ===\n")

    # 1. availability
    ok = is_flash_attn_2_available()
    print(f"is_flash_attn_2_available() = {ok}")
    if not ok:
        sys.exit("FAIL: flash_attn not available")

    # 2. load model with FA2
    print(f"\nLoading Qwen2-VL from {args.model_path} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = model.cuda().eval()

    # 3. config check
    cfg_attn = getattr(model.config, "_attn_implementation", None)
    print(f"model.config._attn_implementation = {cfg_attn}")
    assert cfg_attn == "flash_attention_2", f"Expected flash_attention_2, got {cfg_attn}"

    # 4. vision attention
    visual = model.visual
    vis_blk0 = visual.blocks[0]
    vis_attn = vis_blk0.attn if hasattr(vis_blk0, "attn") else vis_blk0
    vis_cls = type(vis_attn).__name__
    print(f"Vision block attention   = {vis_cls}")
    assert "FlashAttention2" in vis_cls, f"Expected *FlashAttention2*, got {vis_cls}"

    # 5. text attention
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    txt_attn = type(lm.layers[0].self_attn).__name__
    print(f"Text layer attention     = {txt_attn}")
    assert "FlashAttention2" in txt_attn, f"Expected *FlashAttention2*, got {txt_attn}"

    # 6. dtype
    print(f"model dtype              = {model.dtype}")
    print(f"visual dtype             = {next(visual.parameters()).dtype}")

    # 7. verify no SDPA fallback
    all_attn_classes = set()
    for blk in visual.blocks:
        attn = blk.attn if hasattr(blk, "attn") else blk
        all_attn_classes.add(type(attn).__name__)
    for layer in lm.layers:
        all_attn_classes.add(type(layer.self_attn).__name__)
    print(f"All attention classes    = {all_attn_classes}")
    for cls in all_attn_classes:
        assert "SDPA" not in cls, f"SDPA class found: {cls}"

    # 8. cleanup
    del model
    torch.cuda.empty_cache()

    print("\nFLASH-ATTENTION-2 BACKEND CHECK: PASS")
    print("No SDPA fallback detected.")


if __name__ == "__main__":
    main()
