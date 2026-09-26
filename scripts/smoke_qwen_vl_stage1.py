"""One real window through the frozen ViT and Stage-A score backward path."""

import argparse

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from pipeline_stage1 import StreamingVADGenerationModel, _find_embed, _find_visual
from qwen_vl_compat import (
    language_hidden_size,
    load_vl_model,
    load_vl_processor,
    process_video_clips,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    device = torch.device(args.device)
    qwen = load_vl_model(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=None, low_cpu_mem_usage=True,
    ).to(device)
    expected_hidden = language_hidden_size(qwen.config)
    processor = load_vl_processor(args.model_path, min_pixels=100352, max_pixels=100352)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    qwen.config.use_cache = False
    qwen = get_peft_model(qwen, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    for param in _find_visual(qwen).parameters():
        param.requires_grad = False
    model = StreamingVADGenerationModel(
        qwen, d_ssm=256, world_include_decoder=False,
        visual_fusion="state_spatial_film",
    ).to(device)
    for param in model.world_branch.parameters():
        param.requires_grad = False
    model.train()

    reader = VideoReader(args.video_path, ctx=cpu(0))
    indices = list(range(0, min(len(reader), 16 * 3), 3))
    if not indices:
        raise ValueError("empty smoke-test video")
    indices.extend([indices[-1]] * (16 - len(indices)))
    clip = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    processed = process_video_clips(processor, [clip])
    pixels = processed["pixel_values_videos"].to(device)
    grid = processed["video_grid_thw"].to(device)
    assert tuple(grid.shape) == (1, 3) and int(grid[0, 0]) == 8, grid
    valid = torch.ones(1, 1, dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pooled, spatial, spatial_mask = model.extract_window_features(
            pixels, grid, valid, return_spatial=True,
        )
    assert pooled.shape == (1, 1, expected_hidden), pooled.shape
    assert spatial.shape[-1] == expected_hidden, spatial.shape
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        state, _, _, internal, _ = model.encode_window_features(
            pooled, valid, ["smoke"], {}, training=True, return_internal=True,
        )
        prefix, prefix_mask = model.select_visual_prefix(
            state, *valid.nonzero(as_tuple=True), spatial, spatial_mask, internal,
        )
        logits = model.forward_score_visual_prefix(
            prefix, prefix_mask, _find_embed(model.qwen), tokenizer, "Current video status:",
        )
        loss = F.binary_cross_entropy_with_logits(logits.float(), torch.ones_like(logits.float()))
    loss.backward()
    for name, module in (("ssm", model.ssm), ("spatial_film", model.spatial_film), ("score_head", model.score_head)):
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()):
            raise AssertionError(f"{name} has no nonzero gradient")
    print(f"PASS hidden={expected_hidden}, pooled={tuple(pooled.shape)}, "
          f"spatial={tuple(spatial.shape)}, prefix={tuple(prefix.shape)}, "
          f"logits={tuple(logits.shape)}, loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
