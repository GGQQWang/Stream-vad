"""Stage-1 generative semantic alignment training.

Video → frozen ViT → temporal/spatial compress → streaming SSM
→ ViT residual + gated adapter delta → full Qwen2-VL (LoRA, lm_head)
→ <SCORE> token hidden state → score_head → scalar anomaly score.

Three objectives supported:
  - ``score_token``: one-pass LLM forward, extract <SCORE> hidden → MSE(clip_soft)
  - ``answer_ce``: candidate-answer NLL (Normal vs Abnormal) for alignment
  - ``mil_rank``: video-level MIL ranking with language NLL
"""

import argparse
import math
import os
import random
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
from stage1_streaming import (
    collect_summary_triggers,
    dump_window_score_records,
    format_window_score_row,
    make_window_score_record,
    score_bce_loss,
    score_metrics_from_logits,
    sorted_window_score_records,
    summary_ce_loss,
)
from ibq_utils import (
    IBQ_CODE_EMBED_DIM,
    IBQ_CODEBOOK_SIZE,
    IBQTokenCache,
    load_codebook,
    load_ibq_cache,
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


def _add_special_tokens(tokenizer, model) -> Tuple[int, int]:
    """Add ``<SCORE>`` and ``<SUM>`` special tokens and resize embeddings.

    Returns ``(score_token_id, sum_token_id)``.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<SCORE>", "<SUM>"]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    score_id = tokenizer.convert_tokens_to_ids("<SCORE>")
    sum_id = tokenizer.convert_tokens_to_ids("<SUM>")
    if not isinstance(score_id, int) or not isinstance(sum_id, int):
        raise RuntimeError("Failed to add <SCORE> or <SUM> token")
    return score_id, sum_id


# ---------------------------------------------------------------------------
# score-token batch builder
# ---------------------------------------------------------------------------

def build_score_token_batch(
    embed_fn: nn.Module,
    tokenizer,
    state_tokens: torch.Tensor,
    targets: torch.Tensor,
    score_token_id: int,
    prompt_text: str = "Current video status:",
    sum_token_id: int | None = None,
    summary_texts: List[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Build a batch for one-pass anomaly scoring via ``<SCORE>`` token.

    Each sample:  ``[state_token] + prompt_embeds + [<SCORE>_embed]``

    If ``summary_texts`` is provided (last-window-of-clip),
    appends ``[<SUM>] + summary_tokens + [<EOS>]`` after ``<SCORE>``.

    Returns:
        inputs_embeds: ``[N, L, H]``
        attention_mask: ``[N, L]``  bool
        score_mask: ``[N, L]``  bool, True only at ``<SCORE>`` position
        labels: ``[N, L]``  long, -100 everywhere except summary token positions
    """
    device = state_tokens.device
    N, H = state_tokens.shape

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_emb = embed_fn.weight[prompt_ids]                         # [Lp, H]
    score_emb = embed_fn.weight[score_token_id]                      # [H]

    state_part = state_tokens.unsqueeze(1)                            # [N, 1, H]
    prompt_part = prompt_emb.unsqueeze(0).expand(N, -1, -1)            # [N, Lp, H]
    score_part = score_emb.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)  # [N, 1, H]

    has_summary = (
        summary_texts is not None
        and len(summary_texts) > 0
        and sum_token_id is not None
    )
    pre_score_len = 1 + len(prompt_ids)  # state + prompt, <SCORE> is at position pre_score_len

    extra_parts: List[torch.Tensor] = []
    max_text_len = 0
    text_token_lists: List[List[int]] = []
    if has_summary:
        sum_emb = embed_fn.weight[sum_token_id]
        extra_parts.append(sum_emb.unsqueeze(0).unsqueeze(0).expand(N, 1, -1))  # [N, 1, H]
        for st in summary_texts:
            ids = tokenizer.encode(st, add_special_tokens=False)
            ids.append(_find_eos(tokenizer))
            text_token_lists.append(ids)
            max_text_len = max(max_text_len, len(ids))

    embeds = torch.cat([state_part, prompt_part, score_part] + extra_parts, dim=1)
    L_base = embeds.shape[1]

    labels: torch.Tensor
    if has_summary and max_text_len > 0:
        labels = torch.full((N, L_base + max_text_len), -100, dtype=torch.long, device=device)
        text_embeds = torch.zeros(N, max_text_len, H, device=device, dtype=embeds.dtype)
        for i, ids in enumerate(text_token_lists):
            if len(ids) > 0:
                text_embeds[i, :len(ids)] = embed_fn.weight[ids]
                labels[i, L_base:L_base + len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        embeds = torch.cat([embeds, text_embeds], dim=1)
    else:
        labels = torch.full((N, L_base), -100, dtype=torch.long, device=device)

    L = embeds.shape[1]
    attn = torch.ones(N, L, dtype=torch.bool, device=device)
    score_mask = torch.zeros(N, L, dtype=torch.bool, device=device)
    score_mask[:, pre_score_len] = True                              # <SCORE> position

    return {
        "inputs_embeds": embeds,
        "attention_mask": attn,
        "score_mask": score_mask,
        "labels": labels,
    }


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
        self.score_head = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden // 4, 1),
        )
        self.score_query = nn.Parameter(torch.randn(1, llm_hidden) * 0.02)
        self.summary_query = nn.Parameter(torch.randn(1, llm_hidden) * 0.02)
        # world-model auxiliary predictor:
        #   projected causal SSM representation [llm_hidden]  (ssm_out from
        #   SSMBlock.out_proj, which maps the internal Mamba dim to the LLM
        #   hidden size)
        #   -> compact latent [d_ssm]
        #   -> IBQ codebook embedding [IBQ_CODE_EMBED_DIM]
        # Logits are computed as a dot product against the FROZEN IBQ
        # codebook (a non-persistent buffer loaded from the IBQ cache at
        # startup); no trainable [V, H] output layer is needed.
        # Training-only; discarded at inference.
        self.world_predictor = nn.Sequential(
            nn.Linear(llm_hidden, d_ssm),
            nn.GELU(),
            nn.LayerNorm(d_ssm),
            nn.Linear(d_ssm, IBQ_CODE_EMBED_DIM),
        )
        self.register_buffer(
            "ibq_codebook",
            torch.empty(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM),
            persistent=False,
        )
        self.world_codebook_size = IBQ_CODEBOOK_SIZE
        self.llm_hidden = llm_hidden
        self.vit_micro_batch = vit_micro_batch
        self.debug_state = False

        # Full Qwen model with LoRA
        self.qwen = qwen

    def world_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Codebook logits via dot product with the frozen IBQ embedding."""
        expected_dim = self.world_predictor[0].in_features
        if h.shape[-1] != expected_dim:
            raise RuntimeError(
                f"world_predictor input dim mismatch: "
                f"got {h.shape[-1]}, expected {expected_dim}, "
                f"shape={tuple(h.shape)}"
            )
        z = self.world_predictor(h)                     # [..., E]
        if z.shape[-1] != self.ibq_codebook.shape[-1]:
            raise RuntimeError(
                f"IBQ embedding dim mismatch: "
                f"predictor={z.shape[-1]}, "
                f"codebook={self.ibq_codebook.shape[-1]}"
            )
        return F.linear(z, self.ibq_codebook)           # [..., V]

    def encode_window_features(
        self,
        window_batch: torch.Tensor,
        valid_mask: torch.Tensor,
        chunk_video_ids: List[str],
        ssm_state_cache: dict,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """SSM + gated residual over precomputed per-window visual vectors."""
        valid_b, valid_w = valid_mask.nonzero(as_tuple=True)
        ssm_param = next(self.ssm.parameters())
        adapter_param = next(self.adapter.parameters())
        ssm_out = torch.zeros(
            window_batch.shape,
            device=ssm_param.device,
            dtype=ssm_param.dtype,
        )
        for b in range(window_batch.shape[0]):
            vid = chunk_video_ids[b]
            bw = valid_w[valid_b == b]
            if len(bw) == 0:
                continue
            wv = window_batch[b, bw].to(device=ssm_param.device, dtype=ssm_param.dtype).unsqueeze(0)
            prev = ssm_state_cache.get(vid)
            had_prev = prev is not None
            if training and prev is not None:
                prev = {i: s.detach() for i, s in prev.items()}
            if self.debug_state:
                print(
                    f"SSM_STATE video_id={vid} valid_windows={len(bw)} "
                    f"reuse_prev={had_prev} detached={bool(training and had_prev)}"
                )
            out, new_st = self.ssm.forward_chunk(wv, state=prev)
            ssm_out[b, bw] = out.squeeze(0).to(device=ssm_out.device, dtype=ssm_out.dtype)
            ssm_state_cache[vid] = new_st

        ssm_out = ssm_out.to(device=adapter_param.device, dtype=adapter_param.dtype)
        delta = self.adapter(ssm_out)
        alpha = torch.sigmoid(self.alpha_logit).to(device=delta.device, dtype=delta.dtype)
        base = window_batch.to(device=delta.device, dtype=delta.dtype)
        state_embeddings = base + alpha * delta
        return state_embeddings, window_batch, ssm_out, ssm_state_cache

    def extract_window_features(
        self,
        pixel_values: torch.Tensor,
        video_grid_thw: torch.Tensor,
        valid_mask: torch.Tensor,
        return_stats: bool = False,
    ):
        """Frozen ViT + temporal/spatial compression to per-window vectors."""
        B, max_w = valid_mask.shape
        device = valid_mask.device
        n_valid = video_grid_thw.shape[0]
        if n_valid == 0:
            window_batch = torch.zeros(B, max_w, self.llm_hidden, device=device)
            if return_stats:
                return window_batch, {}
            return window_batch

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
        window_batch = torch.zeros(
            B, max_w, self.llm_hidden,
            device=device, dtype=window_vecs.dtype,
        )
        window_batch[valid_b, valid_w] = window_vecs

        if return_stats:
            return window_batch, stats
        return window_batch

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

        if return_stats:
            window_batch, stats = self.extract_window_features(
                pixel_values, video_grid_thw, valid_mask, return_stats=True,
            )
        else:
            window_batch = self.extract_window_features(
                pixel_values, video_grid_thw, valid_mask, return_stats=False,
            )
            stats = {}

        state_embeddings, window_batch, _, ssm_state_cache = self.encode_window_features(
            window_batch, valid_mask, chunk_video_ids, ssm_state_cache, training=training,
        )

        if return_stats:
            return state_embeddings, window_batch, ssm_state_cache, stats
        return state_embeddings, window_batch, ssm_state_cache, None

    def forward_score_token(
        self,
        state_embeddings: torch.Tensor,
        embed_fn: nn.Module,
        tokenizer,
        prompt_text: str,
    ) -> torch.Tensor:
        """One-pass LLM forward -> explicit score query hidden -> anomaly logits."""
        N = state_embeddings.shape[0]
        if N == 0:
            return state_embeddings.new_zeros(0)

        llm_weight = embed_fn.weight
        llm_device = llm_weight.device
        llm_dtype = llm_weight.dtype
        state_embeddings = state_embeddings.to(device=llm_device, dtype=llm_dtype)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=llm_device)
        prompt_emb = embed_fn(prompt_ids_t).unsqueeze(0).expand(N, -1, -1)
        query = self.score_query.to(device=llm_device, dtype=llm_dtype).reshape(1, 1, -1).expand(N, 1, -1)
        inputs = torch.cat([state_embeddings.unsqueeze(1), prompt_emb, query], dim=1)
        if inputs.dtype != llm_dtype or inputs.device != llm_device:
            raise RuntimeError(
                f"Qwen inputs_embeds must match embedding dtype/device: "
                f"got dtype={inputs.dtype}, device={inputs.device}; "
                f"expected dtype={llm_dtype}, device={llm_device}"
            )
        attn = torch.ones(N, inputs.shape[1], dtype=torch.bool, device=llm_device)
        out = self.qwen(
            inputs_embeds=inputs,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[-1][:, -1, :]
        score_param = next(self.score_head.parameters())
        hidden = hidden.to(device=score_param.device, dtype=score_param.dtype)
        return self.score_head(hidden).squeeze(-1)


def _state_dict_shapes_match(module: nn.Module, saved: dict) -> bool:
    """Check a saved state dict against a module BEFORE loading it.

    ``load_state_dict`` may partially copy tensors before raising on a
    shape mismatch; comparing keys and shapes first guarantees the module
    is either fully loadable or left completely untouched.
    """
    current = module.state_dict()
    if current.keys() != saved.keys():
        return False
    for key, tensor in current.items():
        if tuple(tensor.shape) != tuple(saved[key].shape):
            return False
    return True


def _world_model_loss(
    model: StreamingVADGenerationModel,
    ibq_cache: IBQTokenCache,
    batch: dict,
    valid_mask_cpu: torch.Tensor,
    valid_mask: torch.Tensor,
    ssm_out: torch.Tensor,
    horizon: int,
    frame_idx: int,
    detach_states: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Bag-of-tokens IBQ prediction CE from the projected SSM representation.

    For each valid window ``w`` of batch item ``b``: the predictor maps
    ``ssm_out[b, w]`` (the causal SSM state after window ``w``) to a single
    shared distribution over the IBQ codebook, and the loss is the mean CE
    over the tokens of a randomly sampled frame of window
    ``chunk_start + w + horizon``.  Windows without a future target are
    skipped.

    ``detach_states=True`` is used during predictor warmup so no gradient
    flows into the SSM.
    """
    states = ssm_out.detach() if detach_states else ssm_out
    world_logits = model.world_logits(states)                       # [B, max_w, V]
    log_probs = F.log_softmax(world_logits, dim=-1)                 # [B, max_w, V]

    chunk_starts = batch["chunk_start"]
    video_ids = batch["video_id"]
    losses: List[torch.Tensor] = []
    n_windows = 0
    n_tokens = 0
    for b in range(states.shape[0]):
        vid = video_ids[b]
        cs = int(chunk_starts[b])
        for w in range(valid_mask_cpu.shape[1]):
            if not bool(valid_mask_cpu[b, w]):
                continue
            try:
                tgt = ibq_cache.get(vid, cs + w + horizon, frame_idx).long()
            except IndexError:
                continue                                     # no future window
            tgt = tgt.to(device=log_probs.device)
            per_window = -log_probs[b, w].gather(0, tgt).mean()
            losses.append(per_window)
            n_windows += 1
            n_tokens += int(tgt.numel())
    if not losses:
        # no future window has an IBQ target in this batch: return a zero
        # loss that is still connected to the autograd graph so
        # .backward() works when the warmup phase targets loss_world alone
        zero_loss = world_logits.sum() * 0.0
        return zero_loss, {"num_world_windows": 0, "num_world_tokens": 0}
    return torch.stack(losses).mean(), {
        "num_world_windows": n_windows,
        "num_world_tokens": n_tokens,
    }


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
        binary = batch["binary"]
        valid_mask_cpu = batch["valid_mask"]
        valid_mask = valid_mask_cpu.to(device)
        binary = binary.to(device)

        if "features" in batch:
            window_batch = batch["features"].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                state_emb, _, _, ssm_cache = model.encode_window_features(
                    window_batch, valid_mask, batch["video_id"], ssm_cache,
                    training=False,
                )
        else:
            frames_list = batch["frames"]
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

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                state_emb, _, ssm_cache, _ = model.encode_stream(
                    pv, gthw, valid_mask, batch["video_id"], ssm_cache, training=False,
                )

        _clear_finished_states(model, batch, ssm_cache)

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


def _clear_finished_states(model: StreamingVADGenerationModel, batch: dict, ssm_cache: dict) -> None:
    for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
        if is_last:
            existed = vid in ssm_cache
            ssm_cache.pop(vid, None)
            if getattr(model, "debug_state", False):
                print(f"SSM_STATE_CLEAR video_id={vid} existed={existed}")


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
    valid_mask_cpu = batch["valid_mask"]             # [1, max_w]
    valid_mask = valid_mask_cpu.to(device)
    valid_w_cpu = valid_mask_cpu[0].nonzero(as_tuple=True)[0]
    if len(valid_w_cpu) == 0:
        empty = torch.empty(0, model.llm_hidden, device=device)
        return empty, empty.new_empty((0,), dtype=torch.long), empty.new_empty((0,), dtype=torch.long), batch["chunk_start"][0], ssm_cache

    if "features" in batch:
        window_batch = batch["features"].to(device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            state_emb, _, _, ssm_cache = model.encode_window_features(
                window_batch, valid_mask,
                batch["video_id"], ssm_cache,
                training=training,
            )
    else:
        frames = batch["frames"][0]                      # [max_w, F, C, H, W]
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

        _clear_finished_states(model, batch, ssm_cache)

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
    lambda_abnormal: float,
    lambda_sparse: float,
    lambda_rank: float,
    rank_active: bool,
    grad_scale: float,
) -> Dict[str, float]:
    """Second MIL pass for one video.

    The selected ranking term is immediately backpropagated on the video side
    that owns it, so normal and abnormal Qwen graphs are never held together.
    """
    embed_fn = _find_embed(model.qwen)
    ssm_cache: dict = {}
    found_selected = False
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
                found_selected = True
                selected_logit = logits[selected_mask].squeeze(0)
                selected_language = abnormal_language_loss(
                    abnormal_nll, selected_global_idx, global_indices,
                )
                selected_sparse = lambda_sparse * probs[selected_mask].squeeze(0) / max(total_windows, 1)
                selected_loss = selected_sparse + lambda_abnormal * selected_language
                if rank_active:
                    selected_loss = selected_loss - lambda_rank * selected_logit
                (selected_loss / grad_scale).backward()
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
                found_selected = True
                selected_normal_nll = normal_nll[selected_mask].squeeze(0)
                selected_language = lambda_normal * selected_normal_nll / max(total_windows, 1)
                selected_loss = selected_language
                if rank_active:
                    selected_abnormal_nll = _compute_answer_nll(
                        model.qwen, embed_fn, tokenizer, states[selected_mask], 1,
                        prompt_text, normal_answer, abnormal_answer,
                        micro_batch=llm_micro_batch,
                    ).squeeze(0)
                    selected_logit = anomaly_logits_from_nll(
                        selected_normal_nll, selected_abnormal_nll,
                    )
                    selected_loss = selected_loss + lambda_rank * selected_logit
                    metrics["max_score"] = float(selected_logit.detach().item())
                else:
                    metrics["max_score"] = 0.0
                (selected_loss / grad_scale).backward()
                metrics["language_loss"] += float((selected_normal_nll.detach() / max(total_windows, 1)).item())

        _clear_finished_states(model, batch, ssm_cache)

    if not found_selected:
        raise RuntimeError(f"selected window {selected_global_idx} was not found in second MIL pass")
    ssm_cache.clear()
    return metrics


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

        _clear_finished_states(model, batch, ssm_cache)

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
    video_sampler,
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
    was_training = model.training
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
    seen_video_ids: set[str] = set()

    for video_id, sample_indices, video_label in tqdm(
        video_sampler, total=len(video_sampler), desc="Val MIL", leave=False,
    ):
        if video_id in seen_video_ids:
            raise RuntimeError(f"validation video repeated: {video_id}")
        seen_video_ids.add(video_id)

        normal_nll, abnormal_nll, scores, labels = _collect_video_candidate_outputs(
            model, processor, tokenizer, dataset, sample_indices, device, dtype,
            prompt_text, normal_answer, abnormal_answer, llm_micro_batch,
        )

        valid = torch.ones_like(scores, dtype=torch.bool)
        video_max = scores.max()
        if int(video_label) == 0:
            l_normal = normal_language_loss(normal_nll, valid)
            total = lambda_normal * l_normal
            normal_losses.append(float(l_normal.item()))
            normal_max_scores.append(float(video_max.item()))
        else:
            max_idx = torch.argmax(scores)
            l_abnormal = abnormal_nll[max_idx]
            l_sparse = abnormal_sparsity_loss(scores, valid)
            total = lambda_abnormal * l_abnormal + lambda_sparse * l_sparse
            abnormal_losses.append(float(l_abnormal.item()))
            sparse_losses.append(float(l_sparse.item()))
            abnormal_max_scores.append(float(video_max.item()))

        total_losses.append(float(total.item()))
        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(labels.cpu().long().tolist())
        video_scores.append(float(video_max.item()))
        video_labels.append(int(video_label))

    if seen_video_ids != set(video_sampler.video_ids):
        missing = sorted(set(video_sampler.video_ids) - seen_video_ids)
        raise RuntimeError(f"validation videos missing: {missing}")

    for abnormal_score in abnormal_max_scores:
        for normal_score in normal_max_scores:
            rank_losses.append(float(mil_ranking_loss(
                torch.tensor(abnormal_score),
                torch.tensor(normal_score),
                mil_margin,
            ).item()))

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    video_scores_arr = np.array(video_scores)
    video_labels_arr = np.array(video_labels)
    ranking_total = len(abnormal_max_scores) * len(normal_max_scores)
    ranking_hits = [
        1.0 if a > n else 0.0
        for a in abnormal_max_scores
        for n in normal_max_scores
    ]
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
        "n_videos": len(seen_video_ids),
        "n_unique_videos": len(seen_video_ids),
        "ranking_pairs": ranking_total,
    }
    if HAS_SKLEARN and len(set(labels_arr)) > 1:
        metrics["auc"] = roc_auc_score(labels_arr, scores_arr)
        metrics["ap"] = average_precision_score(labels_arr, scores_arr)
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = 0.0
    if HAS_SKLEARN and len(set(video_labels_arr)) > 1:
        metrics["video_auc"] = roc_auc_score(video_labels_arr, video_scores_arr)
        metrics["video_ap"] = average_precision_score(video_labels_arr, video_scores_arr)
    else:
        metrics["video_auc"] = 0.5
        metrics["video_ap"] = 0.0
    metrics["total_loss"] = metrics["total_loss"] + lambda_rank * metrics["ranking_loss"]
    pred = (scores_arr > 0).astype(int)
    metrics["accuracy"] = float((pred == labels_arr).mean()) if len(labels_arr) else 0.0
    model.train(was_training)
    return metrics


# ---------------------------------------------------------------------------
# score-token validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate_score_token(
    model: StreamingVADGenerationModel,
    loader: DataLoader,
    processor: Qwen2VLProcessor,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    score_token_id: int,
    sum_token_id: int | None = None,
    binary_threshold: float = 0.5,
    dump_window_scores: str = "",
) -> dict:
    """One-pass score token evaluation."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_soft_targets: List[torch.Tensor] = []
    window_score_records: List[dict] = []
    ssm_cache: dict = {}
    embed_fn = _find_embed(model.qwen)

    for batch in tqdm(loader, desc="Val score_token", leave=False):
        binary = batch["binary"]
        labels = batch["labels"]
        valid_mask_cpu = batch["valid_mask"]
        valid_mask = valid_mask_cpu.to(device)
        binary = binary.to(device)
        labels = labels.to(device)

        # --- encode ---
        if "features" in batch:
            window_batch = batch["features"].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                state_emb, _, _, ssm_cache = model.encode_window_features(
                    window_batch, valid_mask, batch["video_id"], ssm_cache,
                    training=False,
                )
        else:
            frames_list = batch["frames"]
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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                state_emb, _, ssm_cache, _ = model.encode_stream(
                    pv, gthw, valid_mask, batch["video_id"], ssm_cache, training=False,
                )

        # --- release cache for finished videos ---
        _clear_finished_states(model, batch, ssm_cache)

        # --- score valid windows ---
        valid = valid_mask & (labels >= 0)
        valid_b, valid_w = valid.nonzero(as_tuple=True)
        if len(valid_b) == 0:
            continue
        all_state = state_emb[valid_b, valid_w]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits_flat = model.forward_score_token(
                all_state, embed_fn, tokenizer,
                prompt_text,
            )
        all_logits.append(logits_flat.detach().float().cpu())
        all_soft_targets.append(labels[valid_b, valid_w].detach().float().cpu())
        if dump_window_scores:
            logits_cpu = logits_flat.detach().float().cpu()
            labels_cpu = labels.detach().float().cpu()
            valid_b_cpu = valid_b.detach().cpu()
            valid_w_cpu = valid_w.detach().cpu()
            start_frames = batch["window_start_frames"]
            valid_end_frames = batch["valid_end_frames"]
            for i, (b_t, w_t) in enumerate(zip(valid_b_cpu, valid_w_cpu)):
                b = int(b_t.item())
                w = int(w_t.item())
                window_score_records.append(make_window_score_record(
                    video_id=batch["video_id"][b],
                    window_index=int(batch["chunk_start"][b]) + w,
                    start_frame=int(start_frames[b, w].item()),
                    valid_end_frame=int(valid_end_frames[b, w].item()),
                    fps=float(batch["fps"][b]),
                    soft_target=float(labels_cpu[b, w].item()),
                    score_logit=float(logits_cpu[i].item()),
                    binary_threshold=binary_threshold,
                ))

    model.train()

    if all_logits:
        logits_all = torch.cat(all_logits, dim=0)
        soft_all = torch.cat(all_soft_targets, dim=0)
        valid_all = torch.ones_like(soft_all, dtype=torch.bool)
        metrics = score_metrics_from_logits(
            logits_all, soft_all, valid_all, binary_threshold=binary_threshold,
        )
        try:
            metrics["loss_score"] = float(score_bce_loss(logits_all, soft_all, valid_all).item())
        except ValueError:
            metrics["loss_score"] = math.nan
        metrics["n_samples"] = int(metrics["num_valid_windows"])
    else:
        empty = torch.empty(0)
        metrics = score_metrics_from_logits(empty, empty, torch.empty(0, dtype=torch.bool), binary_threshold)
        metrics["loss_score"] = math.nan
        metrics["n_samples"] = 0

    if math.isnan(float(metrics["auc"])) or math.isnan(float(metrics["ap"])):
        print("WARNING: Validation contains only one binary class; AUC and AP are reported as NaN.")

    if dump_window_scores:
        json_path, csv_path = dump_window_score_records(
            window_score_records,
            dump_window_scores,
            binary_threshold=binary_threshold,
        )
        print(f"Saved validation window scores: json={json_path} csv={csv_path}")
        sorted_records = sorted_window_score_records(
            window_score_records,
            binary_threshold=binary_threshold,
        )
        print("top-10 highest scores")
        for row in sorted_records[:10]:
            print(format_window_score_row(int(row["rank"]), row))
        false_positives = [row for row in sorted_records if row["is_false_positive"]]
        print("top-10 false positives")
        for row in false_positives[:10]:
            print(format_window_score_row(int(row["rank"]), row))
        low_positive = sorted(
            (row for row in sorted_records if int(row["binary_target"]) == 1),
            key=lambda r: float(r["score_prob"]),
        )
        print("top-10 lowest-scoring positive windows")
        for row in low_positive[:10]:
            print(format_window_score_row(int(row["rank"]), row))

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
    parser.add_argument("--anomaly-video-root", default="",
                       help="optional Anomaly-Videos-ALL root; file stems define abnormal video ids for SCORE labels")
    parser.add_argument("--feature-cache-root", default="",
                       help="optional frozen ViT feature cache root; skips video decoding, processor, and ViT")
    parser.add_argument("--log-dir", default="./logs/stage1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--d-ssm", type=int, default=256)
    parser.add_argument("--frames-per-clip", type=int, default=16,
                       help="frames per clip window")
    parser.add_argument("--sample-interval", type=int, default=3,
                       help="stride inside a clip; 3 for ~10fps at 30fps source")
    parser.add_argument("--max-windows", type=int, default=8,
                       help="max scoring windows per TBPTT chunk")
    parser.add_argument("--vit-micro-batch", type=int, default=1)
    parser.add_argument("--llm-micro-batch", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--attn-implementation", type=str, choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--objective", choices=["answer_ce", "mil_rank", "score_token"], default="score_token")
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
    parser.add_argument("--lambda-score", type=float, default=1.0)
    parser.add_argument("--lambda-sum", type=float, default=0.1,
                       help="weight for clip-boundary summary CE loss")
    parser.add_argument("--lambda-world", type=float, default=0.0,
                       help="weight for the world-model IBQ prediction loss (0 disables it)")
    parser.add_argument("--world-horizon", type=int, default=1,
                       help="predict the IBQ tokens of window t+horizon")
    parser.add_argument("--world-warmup-steps", type=int, default=0,
                       help="first N optimizer steps train ONLY the world predictor (h_t detached)")
    parser.add_argument("--ibq-cache-root", default="",
                       help="IBQ token cache root for the world-model loss")
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--dump-window-scores", default="",
                       help="optional JSON path for validation window-level predictions; also writes *_sorted.csv")
    parser.add_argument("--debug-state", action="store_true",
                       help="print SSM state reuse/detach/clear events for streaming checks")
    parser.add_argument("--debug-device", action="store_true",
                       help="print Mamba tensor devices once before the first Triton scan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default="",
                       help="path to a saved checkpoint dir (contains train_state.pt and lora_adapter/); "
                            "training continues from the next epoch with restored optimizer/scheduler")
    args = parser.parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    set_seed(args.seed)
    device = torch.device(args.device)
    # Make the current CUDA device context match args.device.  Triton
    # kernels launch on the *current* device; if tensors live on cuda:N
    # but the context is a different device, kernels fail with
    # "Pointer argument ... cannot be accessed from Triton".
    if device.type == "cuda":
        # normalize "cuda" (no index) to an explicit device so every
        # downstream .to(device) targets a concrete GPU
        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        print(f"Training device: {device}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
    else:
        print(f"Training device: {device} (CPU)")
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
    qwen.config.use_cache = False
    if hasattr(qwen, "gradient_checkpointing_enable"):
        qwen.gradient_checkpointing_enable()
        print("Enabled Qwen gradient checkpointing and disabled use_cache.")
    else:
        print("WARNING: Qwen model does not expose gradient_checkpointing_enable(); use_cache disabled only.")

    _verify_attention_backend(qwen, args.attn_implementation)

    # ---- processor & tokenizer ----
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    score_token_id = None
    sum_token_id = None
    if args.objective == "score_token":
        print("Using explicit trainable score_query and summary_query parameters.")

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0 if args.objective == "mil_rank" else 0.05,
        bias="none", task_type="CAUSAL_LM",
    )
    qwen = get_peft_model(qwen, lora_config)

    # freeze ViT
    for p in _find_visual(qwen).parameters():
        p.requires_grad = False

    model = StreamingVADGenerationModel(
        qwen, d_ssm=args.d_ssm, llm_hidden=qwen.config.hidden_size,
        vit_micro_batch=args.vit_micro_batch,
    ).to(device)
    model.debug_state = bool(args.debug_state)
    model.ssm.debug_device = bool(args.debug_device)

    # one-time world-model shape sanity checks (not per batch)
    assert model.world_predictor[0].in_features == model.llm_hidden, (
        f"world_predictor expects {model.llm_hidden}-dim input, "
        f"got {model.world_predictor[0].in_features}"
    )
    assert model.world_predictor[-1].out_features == IBQ_CODE_EMBED_DIM
    assert model.ibq_codebook.shape == (IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)

    if args.lambda_world <= 0:
        # world model disabled: freeze the predictor so it costs no
        # optimizer memory and receives no gradients (the output layer
        # alone has ~67M parameters)
        for p in model.world_predictor.parameters():
            p.requires_grad = False

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
        total_sampled_frames=args.frames_per_clip, sample_interval=args.sample_interval,
        max_windows=args.max_windows,
        feature_cache_root=args.feature_cache_root or None,
        feature_cache_model_id=args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        anomaly_video_root=args.anomaly_video_root or None,
    )
    from hivau_dataset import hivau_collate
    from hivau_sampler import SequentialVideoSampler, VideoChunkSampler, VideoPairSampler
    train_loader = None
    train_pair_sampler = None
    if args.objective in ("answer_ce", "score_token"):
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
    val_video_sampler = None
    val_ds = None
    if args.val_json:
        val_root = args.val_video_root or args.video_root
        val_ds = HIVAUDataset(
            args.val_json, val_root,
            total_sampled_frames=args.frames_per_clip, sample_interval=args.sample_interval,
            max_windows=args.max_windows,
            feature_cache_root=args.feature_cache_root or None,
            feature_cache_model_id=args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            anomaly_video_root=args.anomaly_video_root or None,
        )
        if args.objective in ("answer_ce", "score_token"):
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=hivau_collate)
        else:
            val_video_sampler = SequentialVideoSampler(val_ds.samples)
            print(
                f"MIL val unique videos: total={len(val_video_sampler)} "
                f"normal={sum(1 for v in val_video_sampler.video_ids if val_video_sampler.video_labels[v] == 0)} "
                f"abnormal={sum(1 for v in val_video_sampler.video_ids if val_video_sampler.video_labels[v] == 1)}"
            )

    # ---- optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    epoch_units = len(train_loader) if args.objective in ("answer_ce", "score_token") else len(train_pair_sampler)
    updates_per_epoch = math.ceil(epoch_units / args.grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = _compute_warmup_steps(total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    embed_fn = _find_embed(model.qwen)

    # ---- world-model IBQ cache (optional) ----
    ibq_cache = None
    if args.lambda_world > 0:
        if not args.ibq_cache_root:
            raise ValueError("--lambda-world > 0 requires --ibq-cache-root")
        probe_video = train_ds.samples[0]["video_id"]
        try:
            probe_meta = load_ibq_cache(args.ibq_cache_root, video_id=probe_video)["metadata"]
        except FileNotFoundError:
            raise ValueError(
                f"IBQ cache missing for {probe_video}; "
                "run precompute_ibq_tokens.py first"
            )
        if int(probe_meta["frames_per_clip"]) != args.frames_per_clip or \
           int(probe_meta["sample_interval"]) != args.sample_interval:
            raise ValueError(
                "IBQ cache windowing mismatch: "
                f"cache={probe_meta['frames_per_clip']}/{probe_meta['sample_interval']}, "
                f"args={args.frames_per_clip}/{args.sample_interval}"
            )
        ibq_cache = IBQTokenCache(args.ibq_cache_root)
        # load the frozen codebook for dot-product logits
        codebook = load_codebook(args.ibq_cache_root).to(
            device=model.ibq_codebook.device, dtype=model.ibq_codebook.dtype,
        )
        if tuple(codebook.shape) != tuple(model.ibq_codebook.shape):
            raise ValueError(
                f"IBQ codebook shape mismatch: {tuple(codebook.shape)} vs "
                f"expected {tuple(model.ibq_codebook.shape)}"
            )
        model.ibq_codebook.copy_(codebook)
        del codebook
        print(f"World-model loss enabled: lambda_world={args.lambda_world}, "
              f"horizon={args.world_horizon}, ibq_cache={args.ibq_cache_root}")

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    best_metric = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            state_path = resume_path / "train_state.pt"
            lora_dir = resume_path / "lora_adapter"
        else:
            state_path = resume_path
            lora_dir = resume_path.parent / "lora_adapter"
        if not state_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {state_path}")
        if not lora_dir.is_dir():
            raise FileNotFoundError(f"resume LoRA adapter not found: {lora_dir}")

        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)

        # config consistency: silently resuming with different data/model
        # config produces garbage — fail loudly instead
        for key in ("frames_per_clip", "sample_interval", "max_windows",
                    "d_ssm", "min_pixels", "max_pixels",
                    "lora_r", "lora_alpha"):
            if key in ckpt and int(ckpt[key]) != int(getattr(args, key)):
                raise ValueError(
                    f"resume config mismatch: {key}={ckpt[key]!r}, "
                    f"expected {getattr(args, key)!r}"
                )
        if "objective" in ckpt and ckpt["objective"] != args.objective:
            raise ValueError(
                f"resume objective mismatch: checkpoint={ckpt['objective']!r}, "
                f"requested {args.objective!r}"
            )
        if "feature_cache_model_id" in ckpt and str(ckpt["feature_cache_model_id"]) != str(args.model_path):
            raise ValueError(
                f"resume model mismatch: checkpoint feature_cache_model_id="
                f"{ckpt['feature_cache_model_id']!r}, expected {args.model_path!r}"
            )

        # model components
        model.ssm.load_state_dict(ckpt["ssm"])
        model.adapter.load_state_dict(ckpt["adapter"])
        world_predictor_compatible = True
        if "world_predictor" in ckpt:
            if _state_dict_shapes_match(
                model.world_predictor, ckpt["world_predictor"],
            ):
                model.world_predictor.load_state_dict(ckpt["world_predictor"])
            else:
                # keep the freshly initialized predictor; never partially
                # load an incompatible state dict
                world_predictor_compatible = False
                print(
                    "WARNING: world_predictor checkpoint is incompatible "
                    "with the current architecture; keeping the newly "
                    "initialized world_predictor."
                )
                if args.lambda_world > 0:
                    print(
                        "WARNING: world_predictor is newly initialized and "
                        "will be trained from scratch while the other "
                        "compatible modules are resumed."
                    )
        else:
            # checkpoint predates the world predictor; the predictor stays
            # fresh.  If it is trainable (lambda_world > 0) the saved
            # optimizer state also lacks its slots.
            if args.lambda_world > 0:
                world_predictor_compatible = False
                print(
                    "WARNING: checkpoint predates world_predictor; the "
                    "predictor is newly initialized and the optimizer state "
                    "will be reset."
                )
        if "score_head" in ckpt:
            model.score_head.load_state_dict(ckpt["score_head"])
        for attr in ("score_query", "summary_query"):
            if attr in ckpt:
                param = getattr(model, attr)
                param.data.copy_(ckpt[attr].to(param.device, param.dtype))
        if "alpha_logit" in ckpt:
            model.alpha_logit.data.copy_(
                ckpt["alpha_logit"].to(model.alpha_logit.device, model.alpha_logit.dtype)
            )

        # LoRA adapter (must stay trainable, otherwise training is a no-op)
        qwen.load_adapter(str(lora_dir), adapter_name="default", is_trainable=True)

        # optimizer / scheduler state: restore only when the trainable
        # parameter set matches the checkpoint, otherwise keep the fresh
        # optimizer/scheduler created above
        optimizer_restored = False
        scheduler_restored = False
        training_state_reset = False
        if "optimizer" in ckpt and world_predictor_compatible:
            optimizer.load_state_dict(ckpt["optimizer"])
            optimizer_restored = True
        elif "optimizer" in ckpt:
            training_state_reset = True
            print(
                "WARNING: optimizer state was not restored because "
                "world_predictor architecture changed. Model weights for "
                "compatible modules were restored, but optimizer moments "
                "are reinitialized."
            )
        if "scheduler" in ckpt and optimizer_restored:
            scheduler.load_state_dict(ckpt["scheduler"])
            scheduler_restored = True
        if training_state_reset:
            print(
                "WARNING: optimizer and scheduler states were reset because "
                "world_predictor architecture changed."
            )

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if start_epoch >= args.epochs:
            raise ValueError(
                f"resume checkpoint is at epoch {ckpt.get('epoch')} but "
                f"--epochs={args.epochs}; pass --epochs greater than the "
                "resumed epoch (total epochs of the whole run)"
            )
        if optimizer_restored:
            global_step = int(ckpt.get("global_step", 0))
        else:
            global_step = 0
            if training_state_reset:
                print(
                    "WARNING: global_step reset to 0 because "
                    "optimizer/scheduler were reinitialized."
                )
        best_metric = float(ckpt.get("best_metric", 0.0))
        print(
            f"Resumed from epoch {start_epoch - 1}: global_step={global_step}, "
            f"best_metric={best_metric:.4f}, "
            f"training epochs {start_epoch}..{args.epochs - 1}"
        )
        if training_state_reset:
            print(
                "Optimizer/scheduler were reset due to world_predictor "
                "architecture change. Training continues from epoch "
                f"{start_epoch} with fresh optimizer state."
            )

    def save_stage1_checkpoint(ckpt_dir: Path, epoch: int) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.qwen.save_pretrained(str(ckpt_dir / "lora_adapter"))
        torch.save({
            "ssm": model.ssm.state_dict(),
            "adapter": model.adapter.state_dict(),
            "score_head": model.score_head.state_dict(),
            "world_predictor": model.world_predictor.state_dict(),
            "score_query": model.score_query.detach().cpu(),
            "summary_query": model.summary_query.detach().cpu(),
            "alpha_logit": model.alpha_logit.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "objective": args.objective,
            "prompt": args.status_prompt,
            "normal_answer": args.normal_answer,
            "abnormal_answer": args.abnormal_answer,
            "supervision_mode": args.supervision_mode,
            "frames_per_clip": args.frames_per_clip,
            "sample_interval": args.sample_interval,
            "max_windows": args.max_windows,
            "d_ssm": args.d_ssm,
            "score_token_id": score_token_id,
            "sum_token_id": sum_token_id,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lambda_normal": args.lambda_normal,
            "lambda_abnormal": args.lambda_abnormal,
            "lambda_rank": args.lambda_rank,
            "lambda_sparse": args.lambda_sparse,
            "lambda_score": args.lambda_score,
            "lambda_sum": args.lambda_sum,
            "lambda_world": args.lambda_world,
            "world_horizon": args.world_horizon,
            "world_warmup_steps": args.world_warmup_steps,
            "ibq_cache_root": args.ibq_cache_root,
            "mil_margin": args.mil_margin,
            "binary_threshold": args.binary_threshold,
            "feature_cache_root": args.feature_cache_root,
            "feature_cache_model_id": args.model_path,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }, str(ckpt_dir / "train_state.pt"))
        processor.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

    # ---- loop ----
    model.train()
    qwen.train()

    for epoch in range(start_epoch, args.epochs):
        train_losses: List[float] = []
        train_normal_losses: List[float] = []
        train_abnormal_losses: List[float] = []
        train_rank_losses: List[float] = []
        train_sparse_losses: List[float] = []
        train_normal_max: List[float] = []
        train_abnormal_max: List[float] = []
        train_ranking_hits: List[float] = []
        train_score_losses: List[float] = []
        train_summary_losses: List[float] = []
        train_world_losses: List[float] = []
        train_valid_windows: List[float] = []
        train_summary_triggers: List[float] = []
        train_skipped_summary: List[float] = []
        train_score_prob_mean: List[float] = []
        train_score_prob_min: List[float] = []
        train_score_prob_max: List[float] = []
        train_soft_target_mean: List[float] = []
        train_valid_windows_total = 0.0
        train_summary_triggers_total = 0.0
        train_skipped_summary_total = 0.0
        train_score_prob_sum = 0.0
        train_soft_target_sum = 0.0
        train_score_prob_min_global = math.inf
        train_score_prob_max_global = -math.inf

        if args.objective == "score_token":
            # ---- fully-supervised score token ----
            ssm_cache: dict = {}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} score_token")

            for step, batch in enumerate(pbar):
                binary = batch["binary"]
                labels = batch["labels"]
                valid_mask_cpu = batch["valid_mask"]
                valid_mask = valid_mask_cpu.to(device)
                binary = binary.to(device)
                labels = labels.to(device)
                B, max_w = binary.shape

                # --- encode ---
                ssm_out = None
                if "features" in batch:
                    window_batch = batch["features"].to(device=device, dtype=dtype)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                        state_emb, visual_windows, ssm_out, ssm_cache = model.encode_window_features(
                            window_batch, valid_mask, batch["video_id"], ssm_cache,
                            training=True,
                        )
                else:
                    frames_list = batch["frames"]
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
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                        state_emb, visual_windows, ssm_cache, _ = model.encode_stream(
                            pv, gthw, valid_mask, batch["video_id"], ssm_cache, training=True,
                        )

                # --- score all valid windows exactly once ---
                valid = valid_mask & (labels >= 0)
                valid_b, valid_w = valid.nonzero(as_tuple=True)
                if len(valid_b) == 0:
                    raise RuntimeError("score_token batch has no valid windows")
                all_state = state_emb[valid_b, valid_w]
                score_logits_flat = model.forward_score_token(
                    all_state, embed_fn, tokenizer, args.status_prompt,
                )
                score_logits = torch.zeros(
                    (B, max_w),
                    device=score_logits_flat.device,
                    dtype=score_logits_flat.dtype,
                )
                score_logits[valid_b, valid_w] = score_logits_flat

                group_start = (step // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(train_loader) - group_start)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                    loss_score = score_bce_loss(
                        score_logits,
                        labels,
                        valid_mask,
                        normalizer=valid_mask.numel(),
                    )

                    triggers, skipped_summary = collect_summary_triggers(batch, valid_mask_cpu)
                    if triggers:
                        trigger_b = torch.tensor([t[0] for t in triggers], dtype=torch.long, device=device)
                        trigger_w = torch.tensor([t[1] for t in triggers], dtype=torch.long, device=device)
                        trigger_states = state_emb[trigger_b, trigger_w]
                        summary_texts = [str(t[2]["text"]) for t in triggers]
                        loss_summary, summary_info = summary_ce_loss(
                            model.qwen, embed_fn, tokenizer,
                            trigger_states, model.summary_query, summary_texts,
                        )
                    else:
                        loss_summary = state_emb.new_zeros(())
                        summary_info = {"num_summary_triggers": 0, "caption_token_count": 0}

                    # --- world-model auxiliary loss (training-only, needs cache mode) ---
                    use_world = (
                        args.lambda_world > 0
                        and ibq_cache is not None
                        and ssm_out is not None
                    )
                    world_info = {"num_world_windows": 0, "num_world_tokens": 0}
                    warmup_phase = False
                    if use_world:
                        warmup_phase = (
                            args.world_warmup_steps > 0
                            and global_step < args.world_warmup_steps
                        )
                        frame_idx = random.randint(0, args.frames_per_clip - 1)
                        loss_world, world_info = _world_model_loss(
                            model, ibq_cache, batch, valid_mask_cpu, valid_mask,
                            ssm_out, args.world_horizon, frame_idx,
                            detach_states=warmup_phase,
                        )
                    else:
                        loss_world = state_emb.new_zeros(())

                    if use_world and warmup_phase:
                        # predictor warmup: only the world predictor trains
                        raw_total_loss = loss_world
                    else:
                        raw_total_loss = (
                            args.lambda_score * loss_score
                            + args.lambda_sum * loss_summary
                            + args.lambda_world * loss_world
                        )
                    total_loss = raw_total_loss / group_size

                if not total_loss.requires_grad:
                    raise RuntimeError(
                        "total_loss unexpectedly has no grad: "
                        f"use_world={use_world}, warmup_phase={warmup_phase}, "
                        f"num_world_windows={world_info['num_world_windows']}, "
                        f"loss_score_requires_grad={loss_score.requires_grad}, "
                        f"loss_summary_requires_grad={loss_summary.requires_grad}, "
                        f"loss_world_requires_grad={loss_world.requires_grad}"
                    )
                total_loss.backward()

                is_update = (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader)
                if is_update:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                train_losses.append(float(raw_total_loss.detach().item()))
                train_score_losses.append(float(loss_score.detach().item()))
                train_summary_losses.append(float(loss_summary.detach().item()))
                train_world_losses.append(float(loss_world.detach().item()))
                num_valid_windows = float(valid.sum().item())
                num_summary_triggers = float(summary_info["num_summary_triggers"])
                train_valid_windows.append(num_valid_windows)
                train_summary_triggers.append(num_summary_triggers)
                train_skipped_summary.append(float(skipped_summary))
                train_valid_windows_total += num_valid_windows
                train_summary_triggers_total += num_summary_triggers
                train_skipped_summary_total += float(skipped_summary)
                with torch.no_grad():
                    valid_probs = torch.sigmoid(score_logits[valid]).detach().float()
                    valid_targets = labels[valid].detach().float()
                    train_score_prob_mean.append(float(valid_probs.mean().item()))
                    train_score_prob_min.append(float(valid_probs.min().item()))
                    train_score_prob_max.append(float(valid_probs.max().item()))
                    train_soft_target_mean.append(float(valid_targets.mean().item()))
                    train_score_prob_sum += float(valid_probs.sum().item())
                    train_soft_target_sum += float(valid_targets.sum().item())
                    train_score_prob_min_global = min(
                        train_score_prob_min_global,
                        float(valid_probs.min().item()),
                    )
                    train_score_prob_max_global = max(
                        train_score_prob_max_global,
                        float(valid_probs.max().item()),
                    )

                _clear_finished_states(model, batch, ssm_cache)

                pbar.set_postfix(
                    loss=sum(train_losses[-10:]) / min(10, len(train_losses)),
                    score=sum(train_score_losses[-10:]) / min(10, len(train_score_losses)),
                    sum_trig=int(train_summary_triggers[-1]),
                )

        elif args.objective == "answer_ce":
            ssm_cache: dict = {}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for step, batch in enumerate(pbar):
                binary = batch["binary"]
                valid_mask_cpu = batch["valid_mask"]
                valid_mask = valid_mask_cpu.to(device)
                binary = binary.to(device)

                if "features" in batch:
                    window_batch = batch["features"].to(device=device, dtype=dtype)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                        state_emb, _, _, ssm_cache = model.encode_window_features(
                            window_batch, valid_mask, batch["video_id"], ssm_cache,
                            training=True,
                        )
                else:
                    frames_list = batch["frames"]
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

                _clear_finished_states(model, batch, ssm_cache)

                pbar.set_postfix(loss=sum(train_losses[-10:]) / min(10, len(train_losses)))
        else:
            pairs = list(train_pair_sampler.iter_epoch(epoch))
            pbar = tqdm(pairs, desc=f"Epoch {epoch} MIL")
            for step, (normal_vid, normal_indices, abnormal_vid, abnormal_indices) in enumerate(pbar):
                group_start = (step // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(pairs) - group_start)

                was_training = model.training
                model.eval()
                try:
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

                    ranking_loss_value = max(
                        0.0,
                        args.mil_margin - abnormal_max_score + normal_max_score,
                    )
                    rank_active = ranking_loss_value > 0.0

                    normal_metrics = _backward_mil_video_pass(
                        model, processor, tokenizer, train_ds, normal_indices,
                        device, dtype,
                        args.status_prompt, args.normal_answer, args.abnormal_answer,
                        args.llm_micro_batch,
                        normal_selected_idx,
                        is_abnormal=False,
                        total_windows=normal_total,
                        lambda_normal=args.lambda_normal,
                        lambda_abnormal=args.lambda_abnormal,
                        lambda_sparse=args.lambda_sparse,
                        lambda_rank=args.lambda_rank,
                        rank_active=rank_active,
                        grad_scale=group_size,
                    )
                    abnormal_metrics = _backward_mil_video_pass(
                        model, processor, tokenizer, train_ds, abnormal_indices,
                        device, dtype,
                        args.status_prompt, args.normal_answer, args.abnormal_answer,
                        args.llm_micro_batch,
                        abnormal_selected_idx,
                        is_abnormal=True,
                        total_windows=abnormal_total,
                        lambda_normal=args.lambda_normal,
                        lambda_abnormal=args.lambda_abnormal,
                        lambda_sparse=args.lambda_sparse,
                        lambda_rank=args.lambda_rank,
                        rank_active=rank_active,
                        grad_scale=group_size,
                    )
                    normal_metrics["max_score"] = normal_max_score
                    abnormal_metrics["max_score"] = abnormal_max_score
                finally:
                    model.train(was_training)

                is_update = (step + 1) % args.grad_accum == 0 or step + 1 == len(pairs)
                if is_update:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                raw_total = (
                    args.lambda_normal * normal_metrics["language_loss"]
                    + args.lambda_abnormal * abnormal_metrics["language_loss"]
                    + args.lambda_rank * ranking_loss_value
                    + abnormal_metrics["sparsity_loss"]
                )
                train_losses.append(raw_total)
                train_normal_losses.append(normal_metrics["language_loss"])
                train_abnormal_losses.append(abnormal_metrics["language_loss"])
                train_rank_losses.append(ranking_loss_value)
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
        if args.objective == "score_token":
            score_prob_mean_epoch = (
                train_score_prob_sum / train_valid_windows_total
                if train_valid_windows_total > 0 else math.nan
            )
            soft_target_mean_epoch = (
                train_soft_target_sum / train_valid_windows_total
                if train_valid_windows_total > 0 else math.nan
            )
            score_prob_min_epoch = (
                train_score_prob_min_global
                if math.isfinite(train_score_prob_min_global) else math.nan
            )
            score_prob_max_epoch = (
                train_score_prob_max_global
                if math.isfinite(train_score_prob_max_global) else math.nan
            )
            writer.add_scalar("train/loss_score", finite_mean(train_score_losses), epoch)
            writer.add_scalar("train/loss_summary", finite_mean(train_summary_losses), epoch)
            if train_world_losses:
                writer.add_scalar("train/loss_world", finite_mean(train_world_losses), epoch)
            writer.add_scalar("train/score_prob_mean", score_prob_mean_epoch, epoch)
            writer.add_scalar("train/score_prob_min", score_prob_min_epoch, epoch)
            writer.add_scalar("train/score_prob_max", score_prob_max_epoch, epoch)
            writer.add_scalar("train/soft_target_mean", soft_target_mean_epoch, epoch)
            writer.add_scalar("train/num_valid_windows", train_valid_windows_total, epoch)
            writer.add_scalar("train/num_summary_triggers", train_summary_triggers_total, epoch)
            writer.add_scalar("train/num_skipped_summary_boundaries", train_skipped_summary_total, epoch)
            print(
                f"  train total={finite_mean(train_losses):.4f} "
                f"score={finite_mean(train_score_losses):.4f} "
                f"summary={finite_mean(train_summary_losses):.4f} "
                f"score_prob={score_prob_mean_epoch:.3f} "
                f"target_mean={soft_target_mean_epoch:.3f} "
                f"valid_windows_total={int(train_valid_windows_total)} "
                f"summary_triggers_total={int(train_summary_triggers_total)} "
                f"skipped_summary_total={int(train_skipped_summary_total)}"
            )
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

        if args.objective == "score_token" and val_loader is not None:
            metrics = validate_score_token(
                model, val_loader, processor, tokenizer, device,
                args.status_prompt, score_token_id, sum_token_id,
                binary_threshold=args.binary_threshold,
                dump_window_scores=args.dump_window_scores,
            )
            print(
                f"  val: loss_score={metrics.get('loss_score', math.nan):.4f} "
                f"mse={metrics.get('mse', math.nan):.4f} mae={metrics.get('mae', math.nan):.4f} "
                f"auc={metrics.get('auc', math.nan):.3f} ap={metrics.get('ap', math.nan):.3f} "
                f"acc={metrics.get('accuracy', 0):.3f} precision={metrics.get('precision', 0):.3f} "
                f"recall={metrics.get('recall', 0):.3f} f1={metrics.get('f1', 0):.3f} "
                f"score_mean={metrics.get('score_mean', math.nan):.3f} "
                f"target_mean={metrics.get('target_mean', math.nan):.3f} "
                f"n={metrics.get('num_valid_windows', 0)}"
            )
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)
            if metrics.get("auc", 0) > best_metric:
                best_metric = metrics["auc"]
                save_stage1_checkpoint(Path(args.log_dir) / "best", epoch)
                print(f"  best auc={best_metric:.4f}")
        elif args.objective == "answer_ce" and val_loader is not None:
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
        elif args.objective == "mil_rank" and val_video_sampler is not None and val_ds is not None:
            metrics = validate_mil_rank(
                model, val_ds, val_video_sampler, processor, tokenizer, device, dtype,
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
