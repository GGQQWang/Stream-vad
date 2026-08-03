"""Stage-1 training pipeline.

Multi-window per video, B×T packed ViT forward, per-clip pool → SSM → LLM.

Architecture:
  [B, T, F, C, H, W]  raw frames (F=20, resolution via min/max_pixels)
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
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from temporal import TemporalTokenReducer
from spatial import SpatialTokenCompressor
from ssm_block import SSMBlock
from vit_forwarder import ViTForwarder
from hivau_dataset import HIVAUDataset


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
    """ViT → spatial → per-clip pool → SSM → LLM → score head."""

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
                            n_layers=n_ssm, llm_hidden=llm_hidden)  # same dim as input
        self.adapter = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
            nn.LayerNorm(llm_hidden),
        )
        self.score_head = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden // 4),
            nn.GELU(),
            nn.Linear(llm_hidden // 4, 1),
        )
        self.llm_hidden = llm_hidden

        # Reference to the LLM transformer (set by trainer after init)
        self.llm: Optional[nn.Module] = None

    def forward(
        self,
        pixel_values: torch.Tensor,              # [total_pixels, 3]  valid clips only
        video_grid_thw: torch.Tensor,            # [n_valid, 3]
        valid_mask: torch.Tensor,                # [B, max_w]  bool
        training: bool = True,
        chunk_video_ids: List[str] | None = None,
        ssm_state_cache: dict | None = None,     # only used when training=False
        return_stats: bool = False,
    ):
        """Returns ``(scores, pooled, ...)``."""
        B, max_w = valid_mask.shape
        device = valid_mask.device

        n_valid = video_grid_thw.shape[0]
        if n_valid == 0:
            scores = torch.zeros(B, max_w, device=device)
            out = (scores, None, ssm_state_cache) if ssm_state_cache is not None else (scores, None)
            if return_stats:
                out = out + ({},)
            return out

        # 1. ViT packed forward
        vit_out = self.vit.forward_batch(
            pixel_values, video_grid_thw, return_stats=return_stats,
        )
        if return_stats:
            tokens, merged_counts, stats = vit_out
        else:
            tokens, merged_counts = vit_out

        # 2. per-clip split + spatial compress + pool → scatter → [B, max_w, 3584]
        tg_list = video_grid_thw[:, 0].tolist()
        clip_token_counts: List[int] = []
        ptr = 0
        for tg in tg_list:
            count = int(merged_counts[ptr: ptr + tg].sum().item())
            clip_token_counts.append(count)
            ptr += tg
        clip_tokens = torch.split(tokens, clip_token_counts, dim=0)
        window_vecs = torch.stack(
            [self.spatial(ct)[0].mean(dim=0) for ct in clip_tokens], dim=0
        )
        valid_b, valid_w = valid_mask.nonzero(as_tuple=True)
        window_batch = torch.zeros(B, max_w, self.llm_hidden, device=device)
        window_batch[valid_b, valid_w] = window_vecs

        # 3. SSM — both paths use forward_chunk() with state carry
        assert chunk_video_ids is not None and ssm_state_cache is not None
        ssm_out = torch.zeros_like(window_batch)
        for b in range(B):
            vid = chunk_video_ids[b]
            bw = valid_w[valid_b == b]
            if len(bw) == 0:
                continue
            wv = window_vecs[valid_b == b].unsqueeze(0)                # [1, T_i, 3584]
            prev = ssm_state_cache.get(vid)
            if training and prev is not None:
                prev = {i: s.detach() for i, s in prev.items()}         # truncated BPTT
            out, new_st = self.ssm.forward_chunk(wv, state=prev)
            ssm_out[b, bw] = out.squeeze(0)
            ssm_state_cache[vid] = new_st

        # 4. semantic-fidelity residual + adapter
        ssm_out = ssm_out + window_batch
        llm_embeds = self.adapter(ssm_out)

        # 5. LLM → score head
        llm_out = self.llm(
            inputs_embeds=llm_embeds,
            attention_mask=valid_mask.long(),
            use_cache=False,
        ).last_hidden_state
        scores = self.score_head(llm_out).squeeze(-1)

        if training:
            if return_stats:
                return scores, window_batch, stats
            return scores, window_batch
        else:
            if return_stats:
                return scores, window_batch, stats, ssm_state_cache
            return scores, window_batch, ssm_state_cache


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: StreamingVADModel,
    loader: DataLoader,
    processor: Qwen2VLProcessor,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict:
    """Return ``{auc, ap, loss, pos_score, neg_score}``."""
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    total_loss = 0.0
    n_valid = 0
    chunk_preds: dict = {}
    expected_windows: dict = {}
    ssm_cache: dict = {}  # val: continuous state, no detach

    for batch in tqdm(loader, desc="Val", leave=False):
        frames_list = batch["frames"]                    # list of [max_w, F, C, H, W]
        binary = batch["binary"]                         # [B, max_w]
        valid_mask_cpu = batch["valid_mask"]              # CPU
        valid_mask = valid_mask_cpu.to(device)

        B, max_w = binary.shape[:2]

        all_clips: List[torch.Tensor] = []
        for b in range(B):
            f = frames_list[b]
            for w in range(max_w):
                if valid_mask_cpu[b, w]:
                    all_clips.append(f[w])

        if not all_clips:
            continue

        processed = processor(
            videos=all_clips,
            return_tensors="pt",
        )
        pixel_vals = processed["pixel_values_videos"].to(device)
        grid_thw = processed["video_grid_thw"].to(device)
        binary = binary.to(device)

        scores, _, ssm_cache = model(pixel_vals, grid_thw, valid_mask,
                                     training=False,
                                     chunk_video_ids=batch["video_id"],
                                     ssm_state_cache=ssm_cache)

        # release cache for finished videos
        for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
            if is_last:
                ssm_cache.pop(vid, None)

        valid = valid_mask & (binary >= 0)
        loss = F.binary_cross_entropy_with_logits(
            scores[valid], binary[valid], pos_weight=pos_weight,
        )
        total_loss += loss.item() * valid.sum().item()
        n_valid += valid.sum().item()

        all_scores.extend(scores[valid].cpu().tolist())
        all_labels.extend(binary[valid].long().cpu().tolist())

        # accumulate per-video predictions for reconstruction
        for b in range(B):
            vid = batch["video_id"][b]
            cs = batch["chunk_start"][b]
            ntw = batch["n_total_windows"][b]
            expected_windows[vid] = ntw
            for w in range(max_w):
                if valid_mask[b, w]:
                    chunk_preds.setdefault(vid, {})[cs + w] = scores[b, w].item()

    # sanity: every video is fully covered (no gaps, no truncation)
    for vid, preds in chunk_preds.items():
        expected = expected_windows[vid]
        assert len(preds) == expected, (
            f"{vid}: predicted {len(preds)} windows, expected {expected}"
        )
        assert set(preds.keys()) == set(range(expected)), (
            f"{vid}: missing windows {set(range(expected)) - set(preds.keys())}"
        )

    model.train()

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    metrics = {"loss": total_loss / max(n_valid, 1)}
    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0
    metrics["pos_score"] = scores_arr[pos_mask].mean().item() if pos_mask.any() else 0.0
    metrics["neg_score"] = scores_arr[neg_mask].mean().item() if neg_mask.any() else 0.0

    if HAS_SKLEARN:
        metrics["auc"] = roc_auc_score(labels_arr, scores_arr) if len(set(labels_arr)) > 1 else 0.5
        metrics["ap"] = average_precision_score(labels_arr, scores_arr) if len(set(labels_arr)) > 1 else 0.0
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = 0.0

    return metrics


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
    parser.add_argument("--max-windows", type=int, default=32,
                       help="max consecutive clips per video sample")
    parser.add_argument("--min-pixels", type=int, default=200704,
                       help="Qwen2-VL min_pixels (448*448=200704)")
    parser.add_argument("--max-pixels", type=int, default=802816,
                       help="Qwen2-VL max_pixels (896*896=802816)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    assert device.type == "cuda", "Stage-1 requires CUDA"
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ---- model ----
    print("Loading Qwen2-VL ...")
    dtype = torch.bfloat16
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype,
        device_map=None, low_cpu_mem_usage=True,
    ).to(device)

    # ---- processor ----
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

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

    # Set LLM transformer (LoRA-aware path so LoRA layers get gradients)
    if hasattr(qwen, "base_model"):
        model.llm = qwen.base_model.model.model   # PeftModel → Qwen2VLModel
    else:
        model.llm = qwen.model.model               # bare Qwen2VLForConditionalGeneration

    # ---- data ----
    train_ds = HIVAUDataset(
        args.train_json, args.video_root,
        total_sampled_frames=args.frames_per_clip,
        sample_interval=1,
        max_windows=args.max_windows,
    )

    from hivau_dataset import hivau_collate  # noqa: E402
    from hivau_sampler import VideoChunkSampler

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=VideoChunkSampler(train_ds.samples, shuffle=True),
        collate_fn=hivau_collate,
    )

    # ---- pos_weight for class imbalance ----
    n_pos = sum(
        sum(b for b in s["clip_bin"]) for s in train_ds.samples
    )
    n_neg = sum(
        sum(1 - b for b in s["clip_bin"]) for s in train_ds.samples
    )
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Pos/Neg windows: {n_pos} / {n_neg}  (ratio {n_pos/max(n_pos+n_neg,1):.3f})  pos_weight={pos_weight.item():.3f}")

    # ---- validation ----
    val_loader = None
    if args.val_json:
        val_ds = HIVAUDataset(
            args.val_json, args.video_root,
            total_sampled_frames=args.frames_per_clip,
            sample_interval=1,
            max_windows=args.max_windows,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=hivau_collate,
        )
        print(f"Val videos: {len(val_ds)}")

    # ---- optimizer (model.parameters() already covers LoRA via model.llm) ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = max(10, int(0.03 * total_steps))          # 3% of total, min 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ---- loop ----
    model.train()
    qwen.train()
    global_step = 0
    best_auc = 0.0

    for epoch in range(args.epochs):
        ssm_cache: dict = {}  # video_id → SSMState (training: detach before forward)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        train_losses: List[float] = []
        train_pos_scores: List[float] = []
        train_neg_scores: List[float] = []
        clip_keep_ratios: List[float] = []
        clip_anomaly_ratios: List[float] = []

        for step, batch in enumerate(pbar):
            frames_list = batch["frames"]                   # list of [max_w, F, C, H, W]
            labels = batch["labels"]                         # [B, max_w] soft
            binary = batch["binary"]                        # [B, max_w] hard
            valid_mask_cpu = batch["valid_mask"]             # [B, max_w] on CPU
            valid_mask = valid_mask_cpu.to(device)

            B, max_w = labels.shape[:2]

            # extract valid clips using CPU mask (no GPU sync per element)
            all_clips: List[torch.Tensor] = []
            for b in range(B):
                f = frames_list[b]
                for w in range(max_w):
                    if valid_mask_cpu[b, w]:
                        all_clips.append(f[w])              # [F, C, H, W]

            if not all_clips:
                continue

            processed = processor(
                videos=all_clips,
                return_tensors="pt",
            )
            pixel_vals = processed["pixel_values_videos"].to(device)
            grid_thw = processed["video_grid_thw"].to(device)
            labels = labels.to(device)
            binary = binary.to(device)

            with torch.autocast(device_type=device.type, dtype=dtype):
                scores, _, stats = model(
                    pixel_vals, grid_thw, valid_mask,
                    training=True, return_stats=True,
                    chunk_video_ids=batch["video_id"],
                    ssm_state_cache=ssm_cache,
                )

                valid = valid_mask & (labels >= 0)
                loss = F.binary_cross_entropy_with_logits(
                    scores[valid], labels[valid], pos_weight=pos_weight,
                )

            group_start = (step // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            raw_loss = loss.detach()
            loss = loss / group_size
            loss.backward()

            is_update = (
                (step + 1) % args.grad_accum == 0
                or step + 1 == len(train_loader)
            )
            if is_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # track per-batch stats
            with torch.no_grad():
                train_losses.append(raw_loss.item())
                vs = scores[valid].cpu()
                bl = binary[valid].cpu()
                pos_m = bl > 0
                if pos_m.any():
                    train_pos_scores.extend(vs[pos_m].tolist())
                if (~pos_m).any():
                    train_neg_scores.extend(vs[~pos_m].tolist())
                # compression stats
                clip_keep_ratios.extend(stats["clip_keep_ratios"])
                clip_anomaly_ratios.extend(labels[valid].cpu().tolist())

            # release SSM cache for finished videos
            for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
                if is_last:
                    ssm_cache.pop(vid, None)

            pbar.set_postfix(
                loss=sum(train_losses[-10:]) / min(10, len(train_losses))
            )

        # ---- end of epoch: log & validate ----
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/loss", np.mean(train_losses), epoch)
        writer.add_scalar("train/lr", lr, epoch)
        if train_pos_scores:
            writer.add_scalar("train/pos_score", np.mean(train_pos_scores), epoch)
        if train_neg_scores:
            writer.add_scalar("train/neg_score", np.mean(train_neg_scores), epoch)

        # compression stats: keep_ratio vs anomaly_ratio
        if clip_keep_ratios:
            keep_arr = np.array(clip_keep_ratios)
            anom_arr = np.array(clip_anomaly_ratios)
            writer.add_scalar("compress/keep_mean", keep_arr.mean(), epoch)

            # per-bucket breakdown (explicit names — no tag collisions)
            buckets = [
                ("0_025",  0.0, 0.25),
                ("025_050", 0.25, 0.5),
                ("050_075", 0.5, 0.75),
                ("075_100", 0.75, 1.01),
            ]
            for tag, lo, hi in buckets:
                m = (anom_arr >= lo) & (anom_arr < hi)
                if m.any():
                    writer.add_scalar(f"compress/keep_anom_{tag}", keep_arr[m].mean(), epoch)

            # normal vs abnormal
            writer.add_scalar("compress/keep_normal", keep_arr[anom_arr == 0].mean() if (anom_arr == 0).any() else 0, epoch)
            writer.add_scalar("compress/keep_abnormal", keep_arr[anom_arr > 0].mean() if (anom_arr > 0).any() else 0, epoch)

        # trainable-weights only (SSM + adapter + score head; ViT frozen, LoRA via PEFT)
        ckpt = {
            "ssm": model.ssm.state_dict(),
            "adapter": model.adapter.state_dict(),
            "score_head": model.score_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_auc": best_auc,
        }

        if val_loader is not None:
            metrics = validate(model, val_loader, processor, device, pos_weight)
            print(f"  val loss={metrics['loss']:.4f}  auc={metrics['auc']:.4f}  "
                  f"ap={metrics['ap']:.4f}  pos={metrics['pos_score']:.3f}  "
                  f"neg={metrics['neg_score']:.3f}")
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                ckpt["best_auc"] = best_auc
                qwen.save_pretrained(os.path.join(args.log_dir, "lora_best"))
                torch.save(ckpt, os.path.join(args.log_dir, "best.pt"))
                print(f"  best auc={best_auc:.4f}")

        if (epoch + 1) % args.save_every == 0:
            qwen.save_pretrained(os.path.join(args.log_dir, f"lora_epoch{epoch}"))
            torch.save(ckpt, os.path.join(args.log_dir, f"ckpt_epoch{epoch}.pt"))

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
