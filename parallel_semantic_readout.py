"""Parallel semantic readout from the existing Stage-1 summary state."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1_streaming import IGNORE_INDEX


class ParallelSemanticReadout(nn.Module):
    """Predict K summary tokens in parallel from one semantic state.

    The learned query embedding at each output position gives positions
    1..K distinct roles; no previous target or predicted token is ever
    fed into another position.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        *,
        num_queries: int = 16,
        decoder_dim: int | None = None,
        num_heads: int = 8,
        num_layers: int = 1,
        use_output_projection: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.num_queries = int(num_queries)
        self.decoder_dim = int(decoder_dim or input_dim)

        self.input_proj = (
            nn.Identity()
            if self.input_dim == self.decoder_dim
            else nn.Linear(self.input_dim, self.decoder_dim)
        )
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.decoder_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.decoder_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.decoder_dim)
        self.output_proj = (
            nn.Linear(self.decoder_dim, self.vocab_size)
            if use_output_projection else None
        )

    def forward(
        self,
        z_sem: torch.Tensor,
        *,
        output_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_sem.ndim != 2:
            raise ValueError(f"z_sem must be [B, D], got {tuple(z_sem.shape)}")
        memory = self.input_proj(z_sem).unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).expand(z_sem.shape[0], -1, -1)
        h = self.decoder(tgt=queries, memory=memory)
        h = self.norm(h)
        if output_weight is not None:
            if output_weight.shape[-1] != h.shape[-1]:
                raise ValueError(
                    f"output_weight hidden dim {output_weight.shape[-1]} "
                    f"does not match decoder dim {h.shape[-1]}"
                )
            return F.linear(h, output_weight.to(device=h.device, dtype=h.dtype))
        if self.output_proj is None:
            raise ValueError("output_weight is required when output_proj is disabled")
        return self.output_proj(h)


def encode_parallel_semantic_targets(
    tokenizer,
    summary_texts: Sequence[str],
    *,
    max_tokens: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    targets = torch.full(
        (len(summary_texts), int(max_tokens)),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    for i, text in enumerate(summary_texts):
        ids = list(tokenizer.encode(str(text), add_special_tokens=False))
        ids = (ids + [int(eos_id)])[: int(max_tokens)]
        if ids:
            targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return targets


def parallel_semantic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, dict]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, K, V], got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(f"target shape {tuple(targets.shape)} does not match logits {tuple(logits.shape)}")
    mask = targets != IGNORE_INDEX
    if not mask.any():
        return logits.sum() * 0.0, {
            "token_accuracy": 0.0,
            "sequence_exact_match": 0.0,
            "eos_accuracy": 0.0,
            "avg_pred_length": 0.0,
            "valid_tokens": 0,
        }
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        token_acc = (pred[mask] == targets[mask]).float().mean()
        seq_ok = (((pred == targets) | ~mask).all(dim=1)).float().mean()
        eos_mask = targets == int(eos_token_id) if eos_token_id is not None else torch.zeros_like(targets, dtype=torch.bool)
        eos_acc = (
            (pred[eos_mask] == targets[eos_mask]).float().mean()
            if eos_mask.any() else torch.tensor(0.0, device=logits.device)
        )
        pred_len = (
            (pred != int(eos_token_id)).float().sum(dim=1).mean()
            if eos_token_id is not None else torch.full((), float(pred.shape[1]), device=logits.device)
        )
    return loss, {
        "token_accuracy": float(token_acc.item()),
        "sequence_exact_match": float(seq_ok.item()),
        "eos_accuracy": float(eos_acc.item()),
        "avg_pred_length": float(pred_len.item()),
        "valid_tokens": int(mask.sum().item()),
    }


def decode_parallel_tokens(tokenizer, token_ids: torch.Tensor) -> list[str]:
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    ids = token_ids.detach().cpu()
    if hasattr(tokenizer, "batch_decode"):
        return list(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return [tokenizer.decode(row.tolist()) for row in ids]


def frozen_lm_head_weight(model: nn.Module, fallback_embed_fn: nn.Module) -> torch.Tensor:
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is not None and hasattr(head, "weight"):
        return head.weight
    return fallback_embed_fn.weight


def freeze_all_except_parallel_readout(model: nn.Module, readout_name: str = "parallel_readout") -> dict:
    total = 0
    trainable = 0
    names: list[str] = []
    for name, param in model.named_parameters():
        total += param.numel()
        allow = name.startswith(readout_name + ".")
        param.requires_grad = allow
        if allow:
            trainable += param.numel()
            names.append(name)
    frozen = total - trainable
    if not names:
        raise RuntimeError(f"no trainable parameters found under {readout_name}")
    bad = [name for name, p in model.named_parameters() if p.requires_grad and not name.startswith(readout_name + ".")]
    if bad:
        raise RuntimeError(f"unexpected trainable parameters outside {readout_name}: {bad}")
    return {
        "total_parameters": total,
        "frozen_parameters": frozen,
        "trainable_parameters": trainable,
        "trainable_parameter_names": names,
    }
