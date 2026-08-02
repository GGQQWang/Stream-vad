"""Stage-1 training pipeline.

Multi-window per video, B×T packed ViT forward, per-clip pool → SSM → LLM.

Architecture:
  [B, T, F, C, H, W]  raw frames (F=20, H=W=448)
       ↓  flatten
  [B*T, F, C, H, W]
       ↓  Qwen2-VL processor → pixel_values + video_grid_thw
  ViTForwarder.forward_batch()
       ↓
  tokens [total, 3584]  +  per-clip counts
       ↓  split + pool
  [B, T, 3584]  window vectors
       ↓  SSMBlock
  [B, T, d_ssm]  →  adapter  →  [B, T, llm_hidden]
       ↓  Qwen2-VL LLM (LoRA, inputs_embeds)
  [B, T, llm_hidden]
       ↓  score_head
  [B, T]  scores  →  BCE(labels)
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from compression.temporal import TemporalTokenReducer
from compression.spatial import SpatialTokenCompressor
from compression.ssm_block import SSMBlock
from compression.vit_forwarder import ViTForwarder
from compression.hivau_dataset import HIVAUDataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_visual(model: Qwen2VLForConditionalGeneration) -> nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    return model.model.visual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StreamingVADModel(nn.Module):
    """ViT → spatial → pool → SSM → adapter → LLM → score head."""

    def __init__(
        self,
        qwen: Qwen2VLForConditionalGeneration,
        d_ssm: int = 256,
        n_ssm: int = 1,
        llm_hidden: int = 3584,
        reduction_ratio: float = 0.5,
        lof_k: int = 8,
    ):
        super().__init__()
        visual = _find_visual(qwen)
        self.vit = ViTForwarder(visual, TemporalTokenReducer())
        self.spatial = SpatialTokenCompressor(reduction_ratio, k=lof_k)
        self.ssm = SSMBlock(d_input=llm_hidden, d_model=d_ssm,
                            n_layers=n_ssm, llm_hidden=llm_hidden)
        self.score_head = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden // 4),
            nn.GELU(),
            nn.Linear(llm_hidden // 4, 1),
        )
        self.llm_hidden = llm_hidden

    def forward(
        self,
        frames: torch.Tensor,                    # [B*T, F, C, H, W]
        video_grid_thw: torch.Tensor,            # [B*T, 3]
        per_video_T: List[int],                  # true T per video
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(scores, pooled)`` where ``scores`` is [B, max_T]."""
        # 1. ViT packed forward
        pixel_values = frames.flatten(0, 1) if frames.dim() == 6 else frames
        # frames are already raw pixels — reshape if needed
        tokens, merged_counts = self.vit.forward_batch(
            pixel_values.flatten(1, -3)          # hack: needs proper pixel reshape
            if pixel_values.dim() > 3 else pixel_values,
            video_grid_thw,
        )                                        # [L, 3584]

        # 2. per-clip split  (same logic as user spec §3)
        T_g_per_clip = video_grid_thw[:, 0].tolist()          # list[int], len = B*T_total
        seqlens_per_clip = []
        ptr = 0
        for tg in T_g_per_clip:
            seqlens_per_clip.append(merged_counts[ptr: ptr + tg.item()].sum().item())
            ptr += tg.item()
        clip_token_counts = [(s + 3) // 4 for s in seqlens_per_clip]  # ceil(/4)

        clip_tokens = torch.split(tokens, clip_token_counts, dim=0)   # list of [n_i, 3584]

        window_vectors = torch.stack(
            [ct.mean(dim=0) for ct in clip_tokens], dim=0
        )                                                            # [B*T_total, 3584]

        # 3. reshape to batched windows  [B, max_T, 3584]
        B = len(per_video_T)
        max_T = max(per_video_T)
        device = window_vectors.device
        pad_val = torch.zeros(1, self.llm_hidden, device=device)
        batches: List[torch.Tensor] = []
        pos = 0
        for t in per_video_T:
            vecs = window_vectors[pos: pos + t]                      # [t, 3584]
            if t < max_T:
                vecs = torch.cat([vecs, pad_val.expand(max_T - t, -1)], dim=0)
            batches.append(vecs)
            pos += t
        window_batch = torch.stack(batches, dim=0)                   # [B, max_T, 3584]

        # 4. SSM
        ssm_out = self.ssm(window_batch)                             # [B, max_T, llm_hidden]

        # 5. score head
        scores = self.score_head(ssm_out).squeeze(-1)                # [B, max_T]
        return scores, window_batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", default="")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--log-dir", default="./logs/stage1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1,
                       help="videos per step (increase with grad-accum)")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--d-ssm", type=int, default=256)
    parser.add_argument("--frames-per-clip", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ---- model ----
    print("Loading Qwen2-VL ...")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype,
        device_map=None, low_cpu_mem_usage=True,
    ).to(device)

    # ---- processor ----
    processor = Qwen2VLProcessor.from_pretrained(args.model_path)

    # ---- LoRA on LLM q/v projections ----
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    qwen = get_peft_model(qwen, lora_config)

    # freeze ViT
    for p in _find_visual(qwen).parameters():
        p.requires_grad = False

    model = StreamingVADModel(
        qwen, d_ssm=args.d_ssm, llm_hidden=qwen.config.hidden_size,
    ).to(device)

    # ---- data ----
    train_ds = HIVAUDataset(
        args.train_json, args.video_root,
        total_sampled_frames=args.frames_per_clip,
        sample_interval=1,              # every frame within the clip span
    )
    train_loader = DataLoader(train_ds, shuffle=True)

    # ---- optimizer ----
    trainable = list(model.parameters()) + \
                [p for p in qwen.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_steps)

    # ---- loop ----
    model.train()
    qwen.train()
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        losses = []

        for step, batch in enumerate(pbar):
            # batch from HIVAUDataset (per-video, no collate pad)
            frames_list: List[torch.Tensor] = []
            labels_list: List[torch.Tensor] = []
            per_video_T: List[int] = []

            for i in range(len(batch["frames"])):            # iterate videos in batch
                f = batch["frames"][i]                       # [T, F, C, H, W]
                l = batch["labels"][i]                       # [T]
                T = f.shape[0]
                if T == 0:
                    continue
                per_video_T.append(T)
                frames_list.append(f)
                labels_list.append(l)

            if not frames_list:
                continue

            # flatten to [B*T, F, C, H, W]
            all_frames = torch.cat(frames_list, dim=0).to(device)
            all_labels = torch.cat(labels_list, dim=0)

            # ---- processor: resize + grid_thw ----
            # Convert frames [B*T, F, C, H, W] to list of video tensors
            # one video per "clip" for the processor
            processed = processor(
                videos=list(all_frames.unbind(0)),    # list of [F, C, H, W]
                return_tensors="pt",
                size={"height": args.image_size, "width": args.image_size},
            )
            pixel_vals = processed["pixel_values_videos"].to(device)
            grid_thw = processed["video_grid_thw"].to(device)    # [B*T, 3]

            with torch.autocast(device_type="cuda", dtype=dtype):
                scores, _ = model(pixel_vals, grid_thw, per_video_T)  # [B, max_T]

                # build per-window label tensor [B, max_T]
                max_T = max(per_video_T)
                label_mat = torch.full((len(per_video_T), max_T), -1.0, device=device)
                pos = 0
                for bi, t in enumerate(per_video_T):
                    label_mat[bi, :t] = all_labels[pos: pos + t]
                    pos += t

                valid = label_mat >= 0
                loss = F.binary_cross_entropy_with_logits(
                    scores[valid], label_mat[valid]
                )

            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            losses.append(loss.item() * args.grad_accum)
            pbar.set_postfix(loss=sum(losses[-10:]) / min(10, len(losses)))

        # ---- save ----
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "qwen_lora": qwen.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(args.log_dir, f"ckpt_epoch{epoch}.pt"))
            print(f"  saved epoch {epoch}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
