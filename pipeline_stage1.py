"""Stage-1 generative semantic alignment training.

Video → frozen ViT → temporal/spatial compress → streaming SSM → adapter
→ state tokens → full Qwen2-VL (LoRA, lm_head) → generate Normal/Abnormal
→ token-level causal LM loss.

This is GENERATIVE alignment, NOT detection.  Detection training
has been moved to pipeline_stage2_detection.py.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    AutoTokenizer,
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


def _find_embed(model: Qwen2VLForConditionalGeneration) -> nn.Module:
    return model.get_input_embeddings()


def _find_eos(tokenizer) -> int:
    """Return the end-of-sequence token id."""
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    # Qwen uses <|im_end|>
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != tokenizer.unk_token_id:
        return im_end
    return tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# generation-batch builder
# ---------------------------------------------------------------------------

def build_status_generation_batch(
    embed_fn: nn.Module,
    tokenizer,
    state_tokens: torch.Tensor,            # [N, H]
    targets: torch.Tensor,                 # [N]  {0,1}
    prompt_text: str = "Current video status:",
    normal_answer: str = "Normal",
    abnormal_answer: str = "Abnormal",
) -> Dict[str, torch.Tensor]:
    """Build a causal-LM batch from SSM state tokens and binary targets.

    Each sample:  [state_token] + prompt_embeds + answer_embeds + eos_embed

    Returns:
        inputs_embeds:    [N, Lmax, H]
        attention_mask:   [N, Lmax]  bool
        labels:           [N, Lmax]  long, -100 on non-answer positions
        answer_token_mask: [N, Lmax]  bool, True on answer+eos positions
    """
    device = state_tokens.device
    N, H = state_tokens.shape

    eos_id = _find_eos(tokenizer)

    # encode answers
    normal_ids = tokenizer.encode(normal_answer, add_special_tokens=False)
    abnormal_ids = tokenizer.encode(abnormal_answer, add_special_tokens=False)
    assert len(normal_ids) >= 1, f"'{normal_answer}' tokenized to empty"
    assert len(abnormal_ids) >= 1, f"'{abnormal_answer}' tokenized to empty"
    assert normal_ids != abnormal_ids, "Normal and Abnormal must differ"

    # answer sequence: answer + eos
    answer_normal = normal_ids + [eos_id]
    answer_abnormal = abnormal_ids + [eos_id]

    # prompt
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # build per-sample
    inputs_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    answer_mask_list: List[torch.Tensor] = []

    embed_weight = embed_fn.weight  # [V, H]

    for i in range(N):
        state = state_tokens[i:i+1]                              # [1, H]

        prompt_emb = embed_weight[prompt_ids]                    # [Lp, H]

        ans_ids = answer_abnormal if targets[i].item() > 0 else answer_normal
        ans_emb = embed_weight[ans_ids]                          # [La, H]

        # concatenate
        inp = torch.cat([state, prompt_emb, ans_emb], dim=0)    # [1+Lp+La, H]

        # labels: -100 for state and prompt, real ids for answer+eos
        lbl = torch.full((inp.shape[0],), -100, dtype=torch.long, device=device)
        lbl[1 + len(prompt_ids):] = torch.tensor(ans_ids, dtype=torch.long, device=device)

        # answer mask
        am = torch.zeros(inp.shape[0], dtype=torch.bool, device=device)
        am[1 + len(prompt_ids):] = True

        attn = torch.ones(inp.shape[0], dtype=torch.bool, device=device)

        inputs_list.append(inp)
        labels_list.append(lbl)
        attn_list.append(attn)
        answer_mask_list.append(am)

    # pad
    max_len = max(t.shape[0] for t in inputs_list)
    inputs_pad = torch.zeros(N, max_len, H, device=device, dtype=state_tokens.dtype)
    labels_pad = torch.full((N, max_len), -100, dtype=torch.long, device=device)
    attn_pad = torch.zeros(N, max_len, dtype=torch.bool, device=device)
    am_pad = torch.zeros(N, max_len, dtype=torch.bool, device=device)

    for i in range(N):
        L = inputs_list[i].shape[0]
        inputs_pad[i, :L] = inputs_list[i]
        labels_pad[i, :L] = labels_list[i]
        attn_pad[i, :L] = attn_list[i]
        am_pad[i, :L] = answer_mask_list[i]

    return {
        "inputs_embeds": inputs_pad,
        "attention_mask": attn_pad,
        "labels": labels_pad,
        "answer_token_mask": am_pad,
        "normal_ids": normal_ids,
        "abnormal_ids": abnormal_ids,
        "eos_id": eos_id,
    }


def masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_token_mask: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    abnormal_loss_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-sample masked token CE with optional abnormal reweighting.

    Args:
        logits:  [N, L, V]
        labels:  [N, L]  (non-answer positions are -100)
        answer_token_mask: [N, L] bool
        targets: [N]  binary {0,1}
        abnormal_loss_weight: optional weight multiplier for abnormal samples.

    Returns:
        loss: scalar
        info: dict with per-sample loss, count, etc.
    """
    N, L, V = logits.shape
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = answer_token_mask[:, 1:].contiguous()

    ce = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(N, -1)                                              # [N, L-1]

    # per-sample: sum CE over answer tokens, divide by answer count
    per_sample = (ce * shift_mask.float()).sum(dim=1) / shift_mask.float().sum(dim=1).clamp_min(1)

    if targets is not None and abnormal_loss_weight != 1.0:
        weights = torch.where(targets > 0, abnormal_loss_weight, 1.0)
        loss = (per_sample * weights.to(per_sample.device)).sum() / weights.sum().clamp_min(1)
    else:
        loss = per_sample.mean()

    info = {
        "loss": loss.item(),
        "n_samples": N,
        "n_answer_tokens": int(shift_mask.sum().item()),
        "mean_ce_per_token": float((ce * shift_mask.float()).sum().item() / shift_mask.float().sum().clamp_min(1).item()),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StreamingVADGenerationModel(nn.Module):
    """ViT → spatial → pool → SSM → adapter → Qwen2-VL (full, LoRA)."""

    def __init__(
        self,
        qwen: Qwen2VLForConditionalGeneration,
        d_ssm: int = 256,
        n_ssm: int = 1,
        llm_hidden: int = 3584,
        reduction_ratio: float = 0.5,
        lof_k: int = 8,
        vit_micro_batch: int = 1,
    ):
        super().__init__()
        visual = _find_visual(qwen)
        self.vit = ViTForwarder(visual, TemporalTokenReducer())
        self.spatial = SpatialTokenCompressor(reduction_ratio, k=lof_k)
        self.ssm = SSMBlock(d_input=llm_hidden, d_model=d_ssm,
                            n_layers=n_ssm, llm_hidden=llm_hidden)
        self.adapter = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
            nn.LayerNorm(llm_hidden),
        )
        self.llm_hidden = llm_hidden
        self.vit_micro_batch = vit_micro_batch

        # Full Qwen model with LoRA
        self.qwen = qwen

    def encode_stream(
        self,
        pixel_values: torch.Tensor,
        video_grid_thw: torch.Tensor,
        valid_mask: torch.Tensor,
        chunk_video_ids: List[str],
        ssm_state_cache: dict,
        training: bool = True,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """ViT → spatial → pool → SSM → adapter → state_embeddings.

        Returns:
            state_embeddings:  [B, max_w, H]
            window_batch:      [B, max_w, H]  pre-adapter
            ssm_state_cache:   updated
            stats (optional)
        """
        B, max_w = valid_mask.shape
        device = valid_mask.device

        n_valid = video_grid_thw.shape[0]
        if n_valid == 0:
            z = torch.zeros(B, max_w, self.llm_hidden, device=device)
            return z, z, ssm_state_cache, ({} if return_stats else None)

        vit_out = self.vit.forward_batch_micro(
            pixel_values, video_grid_thw,
            micro_batch_size=self.vit_micro_batch,
            return_stats=return_stats,
        )
        if return_stats:
            tokens, merged_counts, stats = vit_out
        else:
            tokens, merged_counts = vit_out
            stats = {}

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
        window_batch = torch.zeros(B, max_w, self.llm_hidden, device=device, dtype=window_vecs.dtype)
        window_batch[valid_b, valid_w] = window_vecs

        ssm_out = torch.zeros_like(window_batch)
        for b in range(B):
            vid = chunk_video_ids[b]
            bw = valid_w[valid_b == b]
            if len(bw) == 0:
                continue
            wv = window_vecs[valid_b == b].unsqueeze(0)
            prev = ssm_state_cache.get(vid)
            if training and prev is not None:
                prev = {i: s.detach() for i, s in prev.items()}
            out, new_st = self.ssm.forward_chunk(wv, state=prev)
            ssm_out[b, bw] = out.squeeze(0).to(dtype=ssm_out.dtype)
            ssm_state_cache[vid] = new_st

        ssm_out = ssm_out + window_batch                           # residual
        state_embeddings = self.adapter(ssm_out)                   # [B, max_w, H]

        if return_stats:
            return state_embeddings, window_batch, ssm_state_cache, stats
        return state_embeddings, window_batch, ssm_state_cache, None


def _verify_attention_backend(model: nn.Module, requested: str) -> None:
    """Check that the loaded model actually uses the requested attention backend."""
    print(f"\n--- Attention Backend Check (requested={requested}) ---")
    cfg_attn = getattr(model.config, "_attn_implementation", None)
    print(f"  model.config._attn_implementation = {cfg_attn}")

    from transformers.utils import is_flash_attn_2_available
    fa2_ok = is_flash_attn_2_available()
    print(f"  is_flash_attn_2_available() = {fa2_ok}")

    if requested == "flash_attention_2":
        if not fa2_ok:
            raise RuntimeError("flash_attention_2 requested but not available.")

    visual = _find_visual(model)
    first_vis_blk = visual.blocks[0]
    vis_attn_cls = type(first_vis_blk.attn).__name__ if hasattr(first_vis_blk, "attn") else type(first_vis_blk).__name__
    print(f"  vision block attention class = {vis_attn_cls}")

    if hasattr(model.model, "language_model"):
        lm = model.model.language_model
    else:
        lm = model.model
    first_txt_layer = lm.layers[0]
    txt_attn_cls = type(first_txt_layer.self_attn).__name__
    print(f"  text layer attention class    = {txt_attn_cls}")
    print(f"  model dtype = {model.dtype}")
    print(f"  visual dtype = {next(visual.parameters()).dtype}")

    if requested == "flash_attention_2":
        if "FlashAttention2" not in vis_attn_cls:
            raise RuntimeError(f"Vision attention is {vis_attn_cls}, expected *FlashAttention2*")
        if "FlashAttention2" not in txt_attn_cls:
            raise RuntimeError(f"Text attention is {txt_attn_cls}, expected *FlashAttention2*")
        print("  FLASH-ATTENTION-2 BACKEND CHECK: PASS")
    else:
        print("  SDPA BACKEND CHECK: PASS")
    print("---\n")


# ---------------------------------------------------------------------------
# Validation (candidate NLL)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_generative(
    model: StreamingVADGenerationModel,
    loader: DataLoader,
    processor: Qwen2VLProcessor,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    supervision_mode: str = "all_windows",
    llm_micro_batch: int = 0,
) -> dict:
    """Candidate-answer NLL evaluation.  No score_head."""
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    ssm_cache: dict = {}
    embed_fn = _find_embed(model.qwen)

    for batch in tqdm(loader, desc="Val", leave=False):
        frames_list = batch["frames"]
        binary = batch["binary"]
        valid_mask_cpu = batch["valid_mask"]
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

        processed = processor.image_processor(images=None, videos=all_clips, return_tensors="pt")
        pv = processed["pixel_values_videos"].to(device)
        gthw = processed["video_grid_thw"].to(device)
        binary = binary.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            state_emb, _, ssm_cache, _ = model.encode_stream(
                pv, gthw, valid_mask, batch["video_id"], ssm_cache, training=False,
            )

        for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
            if is_last:
                ssm_cache.pop(vid, None)

        # select state tokens
        valid = valid_mask & (binary >= 0)
        valid_b, valid_w = valid.nonzero(as_tuple=True)
        if len(valid_b) == 0:
            continue

        all_state = state_emb[valid_b, valid_w]                 # [Nv, H]
        all_target = binary[valid_b, valid_w].long()            # [Nv]

        # candidate NLL
        n_normal, n_abnormal = _compute_candidate_nll(
            model.qwen, embed_fn, tokenizer,
            all_state, all_target,
            prompt_text, normal_answer, abnormal_answer,
            micro_batch=llm_micro_batch,
        )
        score = n_abnormal - n_normal                           # higher → more abnormal
        all_scores.extend(score.cpu().tolist())
        all_labels.extend(all_target.cpu().tolist())

    model.train()

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    metrics = {}
    metrics["n_samples"] = len(labels_arr)
    if HAS_SKLEARN and len(set(labels_arr)) > 1:
        metrics["auc"] = roc_auc_score(labels_arr, scores_arr)
        metrics["ap"] = average_precision_score(labels_arr, scores_arr)
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = 0.0

    # accuracy
    pred = (scores_arr > 0).astype(int)
    metrics["accuracy"] = float((pred == labels_arr).mean())
    metrics["normal_recall"] = float((pred[labels_arr == 0] == 0).mean()) if (labels_arr == 0).any() else 0.
    metrics["abnormal_recall"] = float((pred[labels_arr == 1] == 1).mean()) if (labels_arr == 1).any() else 0.

    return metrics


def _compute_candidate_nll(
    qwen, embed_fn, tokenizer,
    state_tokens, targets,
    prompt_text, normal_answer, abnormal_answer,
    micro_batch: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return NLL for Normal and Abnormal for each state token."""
    device = state_tokens.device
    N = state_tokens.shape[0]
    normal_nll = torch.zeros(N, device=device)
    abnormal_nll = torch.zeros(N, device=device)

    for answer, out_tensor in [(normal_answer, normal_nll), (abnormal_answer, abnormal_nll)]:
        batch = build_status_generation_batch(
            embed_fn, tokenizer, state_tokens, torch.zeros(N, device=device, dtype=torch.long),
            prompt_text, normal_answer=answer, abnormal_answer=answer,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = qwen(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                       use_cache=False, return_dict=True)
        logits = out.logits
        shift_logits = logits[:, :-1]
        shift_labels = batch["labels"][:, 1:]
        shift_mask = batch["answer_token_mask"][:, 1:]

        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(N, -1)
        out_tensor.copy_((ce * shift_mask.float()).sum(dim=1) / shift_mask.float().sum(dim=1).clamp_min(1))

    return normal_nll, abnormal_nll


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", default="")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--val-video-root", default="")
    parser.add_argument("--log-dir", default="./logs/stage1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--d-ssm", type=int, default=256)
    parser.add_argument("--frames-per-clip", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--vit-micro-batch", type=int, default=1)
    parser.add_argument("--llm-micro-batch", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--attn-implementation", type=str, choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--supervision-mode", choices=["all_windows", "last_window"], default="all_windows")
    parser.add_argument("--normal-answer", default="Normal")
    parser.add_argument("--abnormal-answer", default="Abnormal")
    parser.add_argument("--status-prompt", default="Current video status:")
    parser.add_argument("--abnormal-loss-weight", type=float, default=1.0)
    parser.add_argument("--generation-examples", type=int, default=0)
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
        attn_implementation=args.attn_implementation,
        device_map=None, low_cpu_mem_usage=True,
    ).to(device)

    _verify_attention_backend(qwen, args.attn_implementation)

    # ---- processor & tokenizer ----
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    qwen = get_peft_model(qwen, lora_config)

    # freeze ViT
    for p in _find_visual(qwen).parameters():
        p.requires_grad = False

    model = StreamingVADGenerationModel(
        qwen, d_ssm=args.d_ssm, llm_hidden=qwen.config.hidden_size,
        vit_micro_batch=args.vit_micro_batch,
    ).to(device)

    # ---- param counts ----
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ssm_params = sum(p.numel() for p in model.ssm.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    lora_params = sum(p.numel() for n, p in qwen.named_parameters() if p.requires_grad and "lora" in n)
    vit_trainable = sum(p.numel() for p in _find_visual(qwen).parameters() if p.requires_grad)
    print(f"Params: total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M  "
          f"SSM={ssm_params/1e3:.0f}K  adapter={adapter_params/1e3:.0f}K  "
          f"LoRA={lora_params/1e3:.0f}K  vit_trainable={vit_trainable}")
    assert vit_trainable == 0, "ViT should be frozen"
    assert ssm_params > 0 and adapter_params > 0 and lora_params > 0

    # ---- data ----
    train_ds = HIVAUDataset(
        args.train_json, args.video_root,
        total_sampled_frames=args.frames_per_clip, sample_interval=1,
        max_windows=args.max_windows,
    )
    from hivau_dataset import hivau_collate
    from hivau_sampler import VideoChunkSampler
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=VideoChunkSampler(train_ds.samples, shuffle=True),
        collate_fn=hivau_collate,
    )

    val_loader = None
    if args.val_json:
        val_root = args.val_video_root or args.video_root
        val_ds = HIVAUDataset(
            args.val_json, val_root,
            total_sampled_frames=args.frames_per_clip, sample_interval=1,
            max_windows=args.max_windows,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=hivau_collate)

    # ---- optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = max(10, int(0.03 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    embed_fn = _find_embed(model.qwen)

    # ---- loop ----
    model.train()
    qwen.train()
    global_step = 0
    best_metric = 0.0

    for epoch in range(args.epochs):
        ssm_cache: dict = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        train_losses: List[float] = []

        for step, batch in enumerate(pbar):
            frames_list = batch["frames"]
            binary = batch["binary"]
            valid_mask_cpu = batch["valid_mask"]
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

            processed = processor.image_processor(images=None, videos=all_clips, return_tensors="pt")
            pv = processed["pixel_values_videos"].to(device)
            gthw = processed["video_grid_thw"].to(device)
            binary = binary.to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                state_emb, _, ssm_cache, _ = model.encode_stream(
                    pv, gthw, valid_mask, batch["video_id"], ssm_cache, training=True,
                )

            # select state tokens
            valid = valid_mask & (binary >= 0)
            valid_b, valid_w = valid.nonzero(as_tuple=True)
            if len(valid_b) == 0:
                continue

            if args.supervision_mode == "last_window":
                # take last valid window per chunk
                keep: List[int] = []
                for b in range(B):
                    bw = valid_w[valid_b == b]
                    if len(bw) > 0:
                        keep.append(int(bw[-1]))
                all_state = state_emb[range(B), torch.tensor(keep, device=device)]
                all_target = binary[range(B), torch.tensor(keep, device=device)].long()
            else:
                all_state = state_emb[valid_b, valid_w]
                all_target = binary[valid_b, valid_w].long()

            # build generation batch
            gen_batch = build_status_generation_batch(
                embed_fn, tokenizer, all_state, all_target,
                prompt_text=args.status_prompt,
                normal_answer=args.normal_answer,
                abnormal_answer=args.abnormal_answer,
            )

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                out = model.qwen(
                    inputs_embeds=gen_batch["inputs_embeds"],
                    attention_mask=gen_batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                loss, loss_info = masked_token_ce(
                    out.logits, gen_batch["labels"], gen_batch["answer_token_mask"],
                    targets=all_target, abnormal_loss_weight=args.abnormal_loss_weight,
                )

            group_start = (step // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            raw_loss = loss.detach()
            loss = loss / group_size
            loss.backward()

            is_update = (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader)
            if is_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            train_losses.append(raw_loss.item())

            for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
                if is_last:
                    ssm_cache.pop(vid, None)

            pbar.set_postfix(loss=sum(train_losses[-10:]) / min(10, len(train_losses)))

        # ---- end of epoch ----
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/loss", np.mean(train_losses), epoch)
        writer.add_scalar("train/lr", lr, epoch)

        # checkpoint
        ckpt_dir = Path(args.log_dir) / f"epoch{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.qwen.save_pretrained(str(ckpt_dir / "lora_adapter"))
        torch.save({
            "ssm": model.ssm.state_dict(),
            "adapter": model.adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "prompt": args.status_prompt,
            "normal_answer": args.normal_answer,
            "abnormal_answer": args.abnormal_answer,
            "supervision_mode": args.supervision_mode,
            "frames_per_clip": args.frames_per_clip,
            "max_windows": args.max_windows,
            "d_ssm": args.d_ssm,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }, str(ckpt_dir / "train_state.pt"))
        processor.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        if val_loader is not None:
            metrics = validate_generative(
                model, val_loader, processor, tokenizer, device,
                args.status_prompt, args.normal_answer, args.abnormal_answer,
                args.supervision_mode, args.llm_micro_batch,
            )
            print(f"  val: acc={metrics.get('accuracy',0):.3f}  auc={metrics.get('auc',0):.3f}  "
                  f"ap={metrics.get('ap',0):.3f}  n_rec={metrics.get('normal_recall',0):.3f}  "
                  f"ab_rec={metrics.get('abnormal_recall',0):.3f}")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)
            if metrics.get("auc", 0) > best_metric:
                best_metric = metrics["auc"]
                model.qwen.save_pretrained(str(Path(args.log_dir) / "lora_best"))
                print(f"  best auc={best_metric:.4f}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
