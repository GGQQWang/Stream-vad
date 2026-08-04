"""Stage-1 generative semantic alignment training.

Video → frozen ViT → temporal/spatial compress → streaming SSM
→ ViT residual + gated adapter delta → full Qwen2-VL (LoRA, lm_head) → generate Normal/Abnormal
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
from mil_utils import (
    abnormal_sparsity_loss,
    anomaly_logits_from_nll,
    anomaly_probs_from_logits,
    finite_mean,
    mil_ranking_loss,
    normal_language_loss,
    select_global_max,
)


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


def _compute_warmup_steps(total_steps: int) -> int:
    if total_steps <= 1:
        return 0
    return min(total_steps - 1, max(1, int(0.03 * total_steps)))


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


def _select_supervised_state_tokens(
    state_emb: torch.Tensor,
    binary: torch.Tensor,
    valid_mask: torch.Tensor,
    supervision_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select window state tokens according to the configured supervision mode."""
    valid = valid_mask & (binary >= 0)
    valid_b, valid_w = valid.nonzero(as_tuple=True)
    if len(valid_b) == 0:
        return state_emb.new_empty((0, state_emb.shape[-1])), binary.new_empty((0,), dtype=torch.long)

    if supervision_mode == "last_window":
        keep_b: List[int] = []
        keep_w: List[int] = []
        for b in range(valid_mask.shape[0]):
            bw = valid_w[valid_b == b]
            if len(bw) > 0:
                keep_b.append(b)
                keep_w.append(int(bw[-1]))
        b_idx = torch.tensor(keep_b, device=state_emb.device, dtype=torch.long)
        w_idx = torch.tensor(keep_w, device=state_emb.device, dtype=torch.long)
        return state_emb[b_idx, w_idx], binary[b_idx, w_idx].long()

    return state_emb[valid_b, valid_w], binary[valid_b, valid_w].long()


def _generation_loss_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_token_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, float]:
    """Return per-sample answer NLLs plus lightweight logging terms."""
    N, _, V = logits.shape
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = answer_token_mask[:, 1:].contiguous()

    ce = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(N, -1)

    mask_f = shift_mask.float()
    per_sample = (ce * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)
    n_answer_tokens = int(shift_mask.sum().item())
    ce_sum = float((ce.detach() * mask_f).sum().item())
    return per_sample, n_answer_tokens, ce_sum


def backward_generation_loss_microbatched(
    qwen,
    embed_fn: nn.Module,
    tokenizer,
    state_tokens: torch.Tensor,
    targets: torch.Tensor,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    abnormal_loss_weight: float,
    micro_batch: int,
    grad_scale: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run Qwen in micro-batches and backprop the globally normalised loss."""
    device = state_tokens.device
    N = state_tokens.shape[0]
    mb = N if micro_batch <= 0 else micro_batch
    weights_all = torch.where(
        targets > 0,
        torch.full_like(targets, abnormal_loss_weight, dtype=torch.float32),
        torch.ones_like(targets, dtype=torch.float32),
    ).to(device)
    denom = weights_all.sum().clamp_min(1)

    total_loss = state_tokens.new_tensor(0.0)
    n_answer_tokens = 0
    ce_sum = 0.0

    for start in range(0, N, mb):
        end = min(start + mb, N)
        state_slice = state_tokens[start:end].detach()
        if state_tokens.requires_grad:
            state_slice.requires_grad_(True)

        gen_batch = build_status_generation_batch(
            embed_fn, tokenizer, state_slice, targets[start:end],
            prompt_text=prompt_text,
            normal_answer=normal_answer,
            abnormal_answer=abnormal_answer,
        )

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            out = qwen(
                inputs_embeds=gen_batch["inputs_embeds"],
                attention_mask=gen_batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            per_sample, n_tokens, batch_ce_sum = _generation_loss_terms(
                out.logits, gen_batch["labels"], gen_batch["answer_token_mask"],
            )
            micro_loss = (per_sample * weights_all[start:end]).sum() / denom

        total_loss = total_loss + micro_loss.detach()
        n_answer_tokens += n_tokens
        ce_sum += batch_ce_sum
        (micro_loss / grad_scale).backward()
        if state_tokens.requires_grad and state_slice.grad is not None:
            state_tokens[start:end].backward(
                state_slice.grad, retain_graph=(end < N),
            )

    info = {
        "loss": total_loss.item(),
        "n_samples": N,
        "n_answer_tokens": n_answer_tokens,
        "mean_ce_per_token": ce_sum / max(n_answer_tokens, 1),
    }
    return total_loss, info


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
        self.alpha_logit = nn.Parameter(torch.tensor(-2.1972246))
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

        delta = self.adapter(ssm_out)
        alpha = torch.sigmoid(self.alpha_logit)
        state_embeddings = window_batch + alpha * delta            # [B, max_w, H]

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

        all_state, all_target = _select_supervised_state_tokens(
            state_emb, binary, valid_mask, supervision_mode,
        )
        if all_state.shape[0] == 0:
            continue

        # candidate NLL
        n_normal, n_abnormal = _compute_candidate_nll(
            model.qwen, embed_fn, tokenizer,
            all_state,
            prompt_text, normal_answer, abnormal_answer,
            micro_batch=llm_micro_batch,
        )
        score = n_normal - n_abnormal                           # higher → more abnormal
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
    state_tokens,
    prompt_text, normal_answer, abnormal_answer,
    micro_batch: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return NLL for Normal and Abnormal for each state token."""
    device = state_tokens.device
    N = state_tokens.shape[0]
    outputs: List[torch.Tensor] = []

    mb = N if micro_batch <= 0 else micro_batch
    for candidate_target in [0, 1]:
        parts: List[torch.Tensor] = []
        for start in range(0, N, mb):
            end = min(start + mb, N)
            candidate_targets = torch.full(
                (end - start,), candidate_target, device=device, dtype=torch.long,
            )
            batch = build_status_generation_batch(
                embed_fn, tokenizer, state_tokens[start:end], candidate_targets,
                prompt_text, normal_answer=normal_answer, abnormal_answer=abnormal_answer,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = qwen(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                           use_cache=False, return_dict=True)
            per_sample, _, _ = _generation_loss_terms(
                out.logits, batch["labels"], batch["answer_token_mask"],
            )
            parts.append(per_sample)
        outputs.append(torch.cat(parts, dim=0))

    return outputs[0], outputs[1]


def _compute_answer_nll(
    qwen, embed_fn, tokenizer,
    state_tokens: torch.Tensor,
    target_value: int,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    micro_batch: int = 0,
) -> torch.Tensor:
    """Return per-window NLL for one fixed candidate answer."""
    device = state_tokens.device
    N = state_tokens.shape[0]
    mb = N if micro_batch <= 0 else micro_batch
    parts: List[torch.Tensor] = []

    for start in range(0, N, mb):
        end = min(start + mb, N)
        targets = torch.full(
            (end - start,), target_value, device=device, dtype=torch.long,
        )
        batch = build_status_generation_batch(
            embed_fn, tokenizer, state_tokens[start:end], targets,
            prompt_text, normal_answer=normal_answer, abnormal_answer=abnormal_answer,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = qwen(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        per_sample, _, _ = _generation_loss_terms(
            out.logits, batch["labels"], batch["answer_token_mask"],
        )
        parts.append(per_sample)

    return torch.cat(parts, dim=0)


def _single_chunk_batch(dataset: HIVAUDataset, sample_idx: int) -> dict:
    from hivau_dataset import hivau_collate

    return hivau_collate([dataset[sample_idx]])


def _encode_chunk_states(
    model: StreamingVADGenerationModel,
    processor: Qwen2VLProcessor,
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
    ssm_cache: dict,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict]:
    """Encode one chunk and return valid state tokens plus global indices."""
    frames = batch["frames"][0]                      # [max_w, F, C, H, W]
    valid_mask_cpu = batch["valid_mask"]             # [1, max_w]
    valid_mask = valid_mask_cpu.to(device)
    valid_w_cpu = valid_mask_cpu[0].nonzero(as_tuple=True)[0]
    if len(valid_w_cpu) == 0:
        empty = torch.empty(0, model.llm_hidden, device=device)
        return empty, empty.new_empty((0,), dtype=torch.long), empty.new_empty((0,), dtype=torch.long), batch["chunk_start"][0], ssm_cache

    all_clips = [frames[int(w)] for w in valid_w_cpu.tolist()]
    processed = processor.image_processor(images=None, videos=all_clips, return_tensors="pt")
    pv = processed["pixel_values_videos"].to(device)
    gthw = processed["video_grid_thw"].to(device)

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
        state_emb, _, ssm_cache, _ = model.encode_stream(
            pv, gthw, valid_mask,
            batch["video_id"], ssm_cache,
            training=training,
        )

    binary = batch["binary"].to(device)
    valid = valid_mask & (binary >= 0)
    valid_b, valid_w = valid.nonzero(as_tuple=True)
    states = state_emb[valid_b, valid_w]
    targets = binary[valid_b, valid_w].long()
    chunk_start = int(batch["chunk_start"][0])
    global_indices = valid_w + chunk_start
    return states, targets, global_indices.long(), chunk_start, ssm_cache


@torch.no_grad()
def _mine_video_max_window(
    model: StreamingVADGenerationModel,
    processor: Qwen2VLProcessor,
    tokenizer,
    dataset: HIVAUDataset,
    sample_indices: List[int],
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    llm_micro_batch: int,
) -> Tuple[int, float]:
    """First MIL pass: find the full-video max anomaly logit without grads."""
    embed_fn = _find_embed(model.qwen)
    ssm_cache: dict = {}
    score_chunks: List[Tuple[int, torch.Tensor]] = []

    for sample_idx in sample_indices:
        batch = _single_chunk_batch(dataset, sample_idx)
        states, _, _, chunk_start, ssm_cache = _encode_chunk_states(
            model, processor, batch, device, dtype, ssm_cache, training=False,
        )
        if states.shape[0] == 0:
            continue
        normal_nll, abnormal_nll = _compute_candidate_nll(
            model.qwen, embed_fn, tokenizer, states,
            prompt_text, normal_answer, abnormal_answer,
            micro_batch=llm_micro_batch,
        )
        score_chunks.append((chunk_start, anomaly_logits_from_nll(normal_nll, abnormal_nll).detach()))

        for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
            if is_last:
                ssm_cache.pop(vid, None)

    selected_idx, selected_score = select_global_max(score_chunks)
    ssm_cache.clear()
    return selected_idx, float(selected_score.item())


def _backward_mil_video_pass(
    model: StreamingVADGenerationModel,
    processor: Qwen2VLProcessor,
    tokenizer,
    dataset: HIVAUDataset,
    sample_indices: List[int],
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    llm_micro_batch: int,
    selected_global_idx: int,
    is_abnormal: bool,
    total_windows: int,
    lambda_normal: float,
    lambda_sparse: float,
    grad_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, Dict[str, float]]:
    """Second MIL pass for one video.

    Returns selected logit, selected direct-language loss if applicable,
    selected sparsity loss if applicable, and detached metrics.
    """
    embed_fn = _find_embed(model.qwen)
    ssm_cache: dict = {}
    selected_logit: torch.Tensor | None = None
    selected_language: torch.Tensor | None = None
    selected_sparse: torch.Tensor | None = None
    metrics: Dict[str, float] = {
        "language_loss": 0.0,
        "sparsity_loss": 0.0,
        "max_score": 0.0,
    }

    for sample_idx in sample_indices:
        batch = _single_chunk_batch(dataset, sample_idx)
        states, _, global_indices, _, ssm_cache = _encode_chunk_states(
            model, processor, batch, device, dtype, ssm_cache, training=True,
        )
        if states.shape[0] == 0:
            continue

        selected_mask = global_indices == int(selected_global_idx)
        nonselected_mask = ~selected_mask

        if is_abnormal:
            normal_nll, abnormal_nll = _compute_candidate_nll(
                model.qwen, embed_fn, tokenizer, states,
                prompt_text, normal_answer, abnormal_answer,
                micro_batch=llm_micro_batch,
            )
            logits = anomaly_logits_from_nll(normal_nll, abnormal_nll)
            probs = anomaly_probs_from_logits(logits)

            if nonselected_mask.any():
                sparse_loss = lambda_sparse * probs[nonselected_mask].sum() / max(total_windows, 1)
                (sparse_loss / grad_scale).backward(retain_graph=bool(selected_mask.any()))
                metrics["sparsity_loss"] += float(sparse_loss.detach().item())

            if selected_mask.any():
                selected_logit = logits[selected_mask].squeeze(0)
                selected_language = abnormal_language_loss(
                    abnormal_nll, selected_global_idx, global_indices,
                )
                selected_sparse = lambda_sparse * probs[selected_mask].squeeze(0) / max(total_windows, 1)
                metrics["language_loss"] = float(selected_language.detach().item())
                metrics["sparsity_loss"] += float(selected_sparse.detach().item())
                metrics["max_score"] = float(selected_logit.detach().item())
        else:
            normal_nll = _compute_answer_nll(
                model.qwen, embed_fn, tokenizer, states, 0,
                prompt_text, normal_answer, abnormal_answer,
                micro_batch=llm_micro_batch,
            )
            if nonselected_mask.any():
                normal_loss = lambda_normal * normal_nll[nonselected_mask].sum() / max(total_windows, 1)
                (normal_loss / grad_scale).backward(retain_graph=bool(selected_mask.any()))
                metrics["language_loss"] += float(normal_loss.detach().item() / max(lambda_normal, 1e-12))

            if selected_mask.any():
                selected_normal_nll = normal_nll[selected_mask].squeeze(0)
                selected_abnormal_nll = _compute_answer_nll(
                    model.qwen, embed_fn, tokenizer, states[selected_mask], 1,
                    prompt_text, normal_answer, abnormal_answer,
                    micro_batch=llm_micro_batch,
                ).squeeze(0)
                selected_logit = anomaly_logits_from_nll(
                    selected_normal_nll, selected_abnormal_nll,
                )
                selected_language = lambda_normal * selected_normal_nll / max(total_windows, 1)
                metrics["language_loss"] += float((selected_normal_nll.detach() / max(total_windows, 1)).item())
                metrics["max_score"] = float(selected_logit.detach().item())

        for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
            if is_last:
                ssm_cache.pop(vid, None)

    if selected_logit is None:
        raise RuntimeError(f"selected window {selected_global_idx} was not found in second MIL pass")
    ssm_cache.clear()
    return selected_logit, selected_language, selected_sparse, metrics


@torch.no_grad()
def _collect_video_candidate_outputs(
    model: StreamingVADGenerationModel,
    processor: Qwen2VLProcessor,
    tokenizer,
    dataset: HIVAUDataset,
    sample_indices: List[int],
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    llm_micro_batch: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_fn = _find_embed(model.qwen)
    ssm_cache: dict = {}
    normal_parts: List[torch.Tensor] = []
    abnormal_parts: List[torch.Tensor] = []
    score_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []

    for sample_idx in sample_indices:
        batch = _single_chunk_batch(dataset, sample_idx)
        states, targets, _, _, ssm_cache = _encode_chunk_states(
            model, processor, batch, device, dtype, ssm_cache, training=False,
        )
        if states.shape[0] > 0:
            normal_nll, abnormal_nll = _compute_candidate_nll(
                model.qwen, embed_fn, tokenizer, states,
                prompt_text, normal_answer, abnormal_answer,
                micro_batch=llm_micro_batch,
            )
            scores = anomaly_logits_from_nll(normal_nll, abnormal_nll)
            normal_parts.append(normal_nll.detach())
            abnormal_parts.append(abnormal_nll.detach())
            score_parts.append(scores.detach())
            label_parts.append(targets.detach())

        for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
            if is_last:
                ssm_cache.pop(vid, None)

    ssm_cache.clear()
    if not score_parts:
        raise RuntimeError("video produced no valid candidate outputs")
    return (
        torch.cat(normal_parts, dim=0),
        torch.cat(abnormal_parts, dim=0),
        torch.cat(score_parts, dim=0),
        torch.cat(label_parts, dim=0),
    )


@torch.no_grad()
def validate_mil_rank(
    model: StreamingVADGenerationModel,
    dataset: HIVAUDataset,
    pair_sampler,
    processor: Qwen2VLProcessor,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    normal_answer: str,
    abnormal_answer: str,
    llm_micro_batch: int,
    lambda_normal: float,
    lambda_abnormal: float,
    lambda_rank: float,
    lambda_sparse: float,
    mil_margin: float,
) -> dict:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    video_scores: List[float] = []
    video_labels: List[int] = []
    total_losses: List[float] = []
    normal_losses: List[float] = []
    abnormal_losses: List[float] = []
    rank_losses: List[float] = []
    sparse_losses: List[float] = []
    normal_max_scores: List[float] = []
    abnormal_max_scores: List[float] = []
    ranking_hits: List[float] = []

    for normal_vid, normal_indices, abnormal_vid, abnormal_indices in tqdm(
        pair_sampler.iter_epoch(0), total=len(pair_sampler), desc="Val MIL", leave=False,
    ):
        n_normal, n_abnormal, n_scores, n_labels = _collect_video_candidate_outputs(
            model, processor, tokenizer, dataset, normal_indices, device, dtype,
            prompt_text, normal_answer, abnormal_answer, llm_micro_batch,
        )
        a_normal, a_abnormal, a_scores, a_labels = _collect_video_candidate_outputs(
            model, processor, tokenizer, dataset, abnormal_indices, device, dtype,
            prompt_text, normal_answer, abnormal_answer, llm_micro_batch,
        )

        n_valid = torch.ones_like(n_scores, dtype=torch.bool)
        a_valid = torch.ones_like(a_scores, dtype=torch.bool)
        l_normal = normal_language_loss(n_normal, n_valid)
        a_idx = torch.argmax(a_scores)
        l_abnormal = a_abnormal[a_idx]
        n_max = n_scores.max()
        a_max = a_scores[a_idx]
        l_rank = mil_ranking_loss(a_max, n_max, mil_margin)
        l_sparse = abnormal_sparsity_loss(a_scores, a_valid)
        total = (
            lambda_normal * l_normal
            + lambda_abnormal * l_abnormal
            + lambda_rank * l_rank
            + lambda_sparse * l_sparse
        )

        total_losses.append(float(total.item()))
        normal_losses.append(float(l_normal.item()))
        abnormal_losses.append(float(l_abnormal.item()))
        rank_losses.append(float(l_rank.item()))
        sparse_losses.append(float(l_sparse.item()))
        normal_max_scores.append(float(n_max.item()))
        abnormal_max_scores.append(float(a_max.item()))
        ranking_hits.append(1.0 if a_max.item() > n_max.item() else 0.0)

        all_scores.extend(n_scores.cpu().tolist())
        all_scores.extend(a_scores.cpu().tolist())
        all_labels.extend(n_labels.cpu().long().tolist())
        all_labels.extend(a_labels.cpu().long().tolist())
        video_scores.extend([float(n_max.item()), float(a_max.item())])
        video_labels.extend([0, 1])

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    video_scores_arr = np.array(video_scores)
    video_labels_arr = np.array(video_labels)
    metrics = {
        "total_loss": finite_mean(total_losses),
        "normal_language_loss": finite_mean(normal_losses),
        "abnormal_language_loss": finite_mean(abnormal_losses),
        "ranking_loss": finite_mean(rank_losses),
        "sparsity_loss": finite_mean(sparse_losses),
        "normal_max_score": finite_mean(normal_max_scores),
        "abnormal_max_score": finite_mean(abnormal_max_scores),
        "ranking_accuracy": finite_mean(ranking_hits),
        "alpha": float(torch.sigmoid(model.alpha_logit).item()),
        "n_samples": len(all_labels),
    }
    if HAS_SKLEARN and len(set(labels_arr)) > 1:
        metrics["auc"] = roc_auc_score(labels_arr, scores_arr)
        metrics["ap"] = average_precision_score(labels_arr, scores_arr)
        metrics["video_auc"] = roc_auc_score(video_labels_arr, video_scores_arr)
        metrics["video_ap"] = average_precision_score(video_labels_arr, video_scores_arr)
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = 0.0
        metrics["video_auc"] = 0.5
        metrics["video_ap"] = 0.0
    pred = (scores_arr > 0).astype(int)
    metrics["accuracy"] = float((pred == labels_arr).mean()) if len(labels_arr) else 0.0
    model.train()
    return metrics


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
    parser.add_argument("--objective", choices=["answer_ce", "mil_rank"], default="answer_ce")
    parser.add_argument("--supervision-mode", choices=["all_windows", "last_window"], default="all_windows")
    parser.add_argument("--normal-answer", default="Normal")
    parser.add_argument("--abnormal-answer", default="Abnormal")
    parser.add_argument("--status-prompt", default="Current video status:")
    parser.add_argument("--abnormal-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-normal", type=float, default=1.0)
    parser.add_argument("--lambda-abnormal", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-sparse", type=float, default=1e-3)
    parser.add_argument("--mil-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    set_seed(args.seed)
    device = torch.device(args.device)
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
    from hivau_sampler import VideoChunkSampler, VideoPairSampler
    train_loader = None
    train_pair_sampler = None
    if args.objective == "answer_ce":
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=VideoChunkSampler(train_ds.samples, shuffle=True),
            collate_fn=hivau_collate,
        )
    else:
        train_pair_sampler = VideoPairSampler(train_ds.samples, shuffle=True, seed=args.seed)
        print(
            f"MIL train videos: normal={len(train_pair_sampler.normal_videos)} "
            f"abnormal={len(train_pair_sampler.abnormal_videos)} "
            f"pairs_per_epoch={len(train_pair_sampler)}"
        )

    val_loader = None
    val_pair_sampler = None
    val_ds = None
    if args.val_json:
        val_root = args.val_video_root or args.video_root
        val_ds = HIVAUDataset(
            args.val_json, val_root,
            total_sampled_frames=args.frames_per_clip, sample_interval=1,
            max_windows=args.max_windows,
        )
        if args.objective == "answer_ce":
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=hivau_collate)
        else:
            val_pair_sampler = VideoPairSampler(val_ds.samples, shuffle=False, seed=args.seed)
            print(
                f"MIL val videos: normal={len(val_pair_sampler.normal_videos)} "
                f"abnormal={len(val_pair_sampler.abnormal_videos)} "
                f"pairs={len(val_pair_sampler)}"
            )

    # ---- optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    epoch_units = len(train_loader) if args.objective == "answer_ce" else len(train_pair_sampler)
    updates_per_epoch = math.ceil(epoch_units / args.grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = _compute_warmup_steps(total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    embed_fn = _find_embed(model.qwen)

    def save_stage1_checkpoint(ckpt_dir: Path, epoch: int) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.qwen.save_pretrained(str(ckpt_dir / "lora_adapter"))
        torch.save({
            "ssm": model.ssm.state_dict(),
            "adapter": model.adapter.state_dict(),
            "alpha_logit": model.alpha_logit.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "objective": args.objective,
            "prompt": args.status_prompt,
            "normal_answer": args.normal_answer,
            "abnormal_answer": args.abnormal_answer,
            "supervision_mode": args.supervision_mode,
            "frames_per_clip": args.frames_per_clip,
            "max_windows": args.max_windows,
            "d_ssm": args.d_ssm,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lambda_normal": args.lambda_normal,
            "lambda_abnormal": args.lambda_abnormal,
            "lambda_rank": args.lambda_rank,
            "lambda_sparse": args.lambda_sparse,
            "mil_margin": args.mil_margin,
        }, str(ckpt_dir / "train_state.pt"))
        processor.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

    # ---- loop ----
    model.train()
    qwen.train()
    global_step = 0
    best_metric = 0.0

    for epoch in range(args.epochs):
        train_losses: List[float] = []
        train_normal_losses: List[float] = []
        train_abnormal_losses: List[float] = []
        train_rank_losses: List[float] = []
        train_sparse_losses: List[float] = []
        train_normal_max: List[float] = []
        train_abnormal_max: List[float] = []
        train_ranking_hits: List[float] = []

        if args.objective == "answer_ce":
            ssm_cache: dict = {}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

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

                all_state, all_target = _select_supervised_state_tokens(
                    state_emb, binary, valid_mask, args.supervision_mode,
                )
                if all_state.shape[0] == 0:
                    continue

                group_start = (step // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(train_loader) - group_start)
                raw_loss, loss_info = backward_generation_loss_microbatched(
                    model.qwen, embed_fn, tokenizer,
                    all_state, all_target,
                    prompt_text=args.status_prompt,
                    normal_answer=args.normal_answer,
                    abnormal_answer=args.abnormal_answer,
                    abnormal_loss_weight=args.abnormal_loss_weight,
                    micro_batch=args.llm_micro_batch,
                    grad_scale=group_size,
                    dtype=dtype,
                )

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
        else:
            pairs = list(train_pair_sampler.iter_epoch(epoch))
            pbar = tqdm(pairs, desc=f"Epoch {epoch} MIL")
            for step, (normal_vid, normal_indices, abnormal_vid, abnormal_indices) in enumerate(pbar):
                group_start = (step // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(pairs) - group_start)

                normal_selected_idx, normal_max_score = _mine_video_max_window(
                    model, processor, tokenizer, train_ds, normal_indices,
                    device, dtype,
                    args.status_prompt, args.normal_answer, args.abnormal_answer,
                    args.llm_micro_batch,
                )
                abnormal_selected_idx, abnormal_max_score = _mine_video_max_window(
                    model, processor, tokenizer, train_ds, abnormal_indices,
                    device, dtype,
                    args.status_prompt, args.normal_answer, args.abnormal_answer,
                    args.llm_micro_batch,
                )

                normal_total = sum(
                    int(train_ds.samples[i]["chunk_end"]) - int(train_ds.samples[i]["chunk_start"])
                    for i in normal_indices
                )
                abnormal_total = sum(
                    int(train_ds.samples[i]["chunk_end"]) - int(train_ds.samples[i]["chunk_start"])
                    for i in abnormal_indices
                )

                normal_logit, normal_selected_loss, _, normal_metrics = _backward_mil_video_pass(
                    model, processor, tokenizer, train_ds, normal_indices,
                    device, dtype,
                    args.status_prompt, args.normal_answer, args.abnormal_answer,
                    args.llm_micro_batch,
                    normal_selected_idx,
                    is_abnormal=False,
                    total_windows=normal_total,
                    lambda_normal=args.lambda_normal,
                    lambda_sparse=args.lambda_sparse,
                    grad_scale=group_size,
                )
                abnormal_logit, abnormal_selected_loss, abnormal_selected_sparse, abnormal_metrics = _backward_mil_video_pass(
                    model, processor, tokenizer, train_ds, abnormal_indices,
                    device, dtype,
                    args.status_prompt, args.normal_answer, args.abnormal_answer,
                    args.llm_micro_batch,
                    abnormal_selected_idx,
                    is_abnormal=True,
                    total_windows=abnormal_total,
                    lambda_normal=args.lambda_normal,
                    lambda_sparse=args.lambda_sparse,
                    grad_scale=group_size,
                )

                ranking_loss = args.lambda_rank * mil_ranking_loss(
                    abnormal_logit, normal_logit, args.mil_margin,
                )
                final_loss = ranking_loss
                if normal_selected_loss is not None:
                    final_loss = final_loss + normal_selected_loss
                if abnormal_selected_loss is not None:
                    final_loss = final_loss + args.lambda_abnormal * abnormal_selected_loss
                if abnormal_selected_sparse is not None:
                    final_loss = final_loss + abnormal_selected_sparse
                (final_loss / group_size).backward()

                is_update = (step + 1) % args.grad_accum == 0 or step + 1 == len(pairs)
                if is_update:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                raw_total = (
                    args.lambda_normal * normal_metrics["language_loss"]
                    + args.lambda_abnormal * abnormal_metrics["language_loss"]
                    + float(ranking_loss.detach().item())
                    + abnormal_metrics["sparsity_loss"]
                )
                train_losses.append(raw_total)
                train_normal_losses.append(normal_metrics["language_loss"])
                train_abnormal_losses.append(abnormal_metrics["language_loss"])
                train_rank_losses.append(float((ranking_loss.detach() / max(args.lambda_rank, 1e-12)).item()))
                train_sparse_losses.append(abnormal_metrics["sparsity_loss"] / max(args.lambda_sparse, 1e-12))
                train_normal_max.append(normal_metrics["max_score"])
                train_abnormal_max.append(abnormal_metrics["max_score"])
                train_ranking_hits.append(1.0 if abnormal_max_score > normal_max_score else 0.0)

                pbar.set_postfix(
                    loss=sum(train_losses[-10:]) / min(10, len(train_losses)),
                    n=normal_vid,
                    a=abnormal_vid,
                )

        # ---- end of epoch ----
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/loss", np.mean(train_losses), epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/alpha", torch.sigmoid(model.alpha_logit).item(), epoch)
        if args.objective == "mil_rank":
            writer.add_scalar("train/normal_language_loss", finite_mean(train_normal_losses), epoch)
            writer.add_scalar("train/abnormal_language_loss", finite_mean(train_abnormal_losses), epoch)
            writer.add_scalar("train/ranking_loss", finite_mean(train_rank_losses), epoch)
            writer.add_scalar("train/sparsity_loss", finite_mean(train_sparse_losses), epoch)
            writer.add_scalar("train/normal_max_score", finite_mean(train_normal_max), epoch)
            writer.add_scalar("train/abnormal_max_score", finite_mean(train_abnormal_max), epoch)
            writer.add_scalar("train/ranking_accuracy", finite_mean(train_ranking_hits), epoch)

        if (epoch + 1) % args.save_every == 0:
            save_stage1_checkpoint(Path(args.log_dir) / f"epoch{epoch}", epoch)

        if args.objective == "answer_ce" and val_loader is not None:
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
                save_stage1_checkpoint(Path(args.log_dir) / "best", epoch)
                print(f"  best auc={best_metric:.4f}")
        elif args.objective == "mil_rank" and val_pair_sampler is not None and val_ds is not None:
            metrics = validate_mil_rank(
                model, val_ds, val_pair_sampler, processor, tokenizer, device, dtype,
                args.status_prompt, args.normal_answer, args.abnormal_answer,
                args.llm_micro_batch,
                args.lambda_normal, args.lambda_abnormal, args.lambda_rank,
                args.lambda_sparse, args.mil_margin,
            )
            print(
                f"  val total={metrics['total_loss']:.4f} normal={metrics['normal_language_loss']:.4f} "
                f"abnormal={metrics['abnormal_language_loss']:.4f} rank={metrics['ranking_loss']:.4f} "
                f"sparse={metrics['sparsity_loss']:.4f} n_max={metrics['normal_max_score']:.3f} "
                f"a_max={metrics['abnormal_max_score']:.3f} rank_acc={metrics['ranking_accuracy']:.3f} "
                f"alpha={metrics['alpha']:.3f} auc={metrics['auc']:.3f} video_auc={metrics['video_auc']:.3f}"
            )
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)
            if metrics.get("video_auc", metrics.get("auc", 0)) > best_metric:
                best_metric = metrics.get("video_auc", metrics.get("auc", 0))
                save_stage1_checkpoint(Path(args.log_dir) / "best", epoch)
                print(f"  best video_auc={best_metric:.4f}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
