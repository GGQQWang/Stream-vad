"""CPU-testable Stage-1 streaming objectives and metadata helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass(frozen=True)
class WindowInfo:
    index: int
    start_frame: int
    end_frame: int
    valid_end_frame: int
    sampled_frames: tuple[int, ...]
    soft_label: float
    summary_triggers: tuple[dict, ...]
    skipped_summary_boundaries: int = 0


def sampled_fps(source_fps: float, sample_interval: int) -> float:
    return float(source_fps) / float(sample_interval)


def window_span_frames(frames_per_clip: int, sample_interval: int) -> int:
    return int(frames_per_clip) * int(sample_interval)


def build_window_infos(
    *,
    n_frames: int,
    fps: float,
    events: Sequence[Sequence[float]],
    frames_per_clip: int,
    sample_interval: int,
    summary_clips: Sequence[dict] | None = None,
) -> List[WindowInfo]:
    """Build window labels/triggers without using future frames.

    ``frames_per_clip`` is the number of sampled frames in one scoring window.
    The source-frame span is therefore ``frames_per_clip * sample_interval``.
    Summary clips are triggered only when their end frame exactly matches a
    window's valid end frame.  A clip ending inside a window is skipped so the
    summary loss cannot see frames after the clip boundary.
    """
    if n_frames <= 0:
        return []
    span = window_span_frames(frames_per_clip, sample_interval)
    n_windows = math.ceil(n_frames / span)

    abnormal = torch.zeros(n_frames, dtype=torch.bool)
    for start_sec, end_sec in events:
        lo = max(0, int(float(start_sec) * fps))
        hi = min(n_frames, int(float(end_sec) * fps))
        if hi > lo:
            abnormal[lo:hi] = True

    clips_by_end: dict[int, List[dict]] = {}
    skipped_by_window = [0 for _ in range(n_windows)]
    for clip in summary_clips or []:
        clip_id = clip.get("clip_id", f"clip_{len(clips_by_end)}")
        start_frame = max(0, int(clip["clip_start_frame"]))
        end_frame = min(n_frames, int(clip["clip_end_frame"]))
        text = str(clip.get("text", ""))
        if end_frame <= start_frame or not text:
            continue
        wi = min(max(math.ceil(end_frame / span) - 1, 0), n_windows - 1)
        win_start = wi * span
        win_valid_end = min(win_start + span, n_frames)
        if end_frame == win_valid_end:
            clips_by_end.setdefault(end_frame, []).append({
                "clip_id": clip_id,
                "clip_start_frame": start_frame,
                "clip_end_frame": end_frame,
                "text": text,
            })
        elif win_start < end_frame < win_valid_end:
            skipped_by_window[wi] += 1

    infos: List[WindowInfo] = []
    for wi in range(n_windows):
        start = wi * span
        valid_end = min(start + span, n_frames)
        sampled = tuple(range(start, valid_end, sample_interval))
        if not sampled:
            soft = 0.0
        else:
            soft = float(abnormal[list(sampled)].float().mean().item())
        triggers = tuple(clips_by_end.get(valid_end, []))
        infos.append(WindowInfo(
            index=wi,
            start_frame=start,
            end_frame=start + span,
            valid_end_frame=valid_end,
            sampled_frames=sampled,
            soft_label=soft,
            summary_triggers=triggers,
            skipped_summary_boundaries=skipped_by_window[wi],
        ))
    return infos


def score_bce_loss(
    score_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask & (soft_targets >= 0)
    if not valid.any():
        raise ValueError("score loss has no valid windows")
    return F.binary_cross_entropy_with_logits(score_logits[valid], soft_targets[valid].float())


def score_metrics_from_logits(
    score_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    binary_threshold: float = 0.5,
) -> dict:
    valid = valid_mask & (soft_targets >= 0)
    probs = torch.sigmoid(score_logits[valid]).detach().float()
    soft = soft_targets[valid].detach().float()
    binary = (soft >= binary_threshold).long()
    if probs.numel() == 0:
        return {
            "score_prob": probs,
            "soft_targets": soft,
            "binary_targets": binary,
            "mse": math.nan,
            "mae": math.nan,
        }
    return {
        "score_prob": probs,
        "soft_targets": soft,
        "binary_targets": binary,
        "mse": float(F.mse_loss(probs, soft).item()),
        "mae": float(F.l1_loss(probs, soft).item()),
    }


def future_prediction_loss(
    states: torch.Tensor,
    visual_windows: torch.Tensor,
    valid_mask: torch.Tensor,
    window_indices: torch.Tensor,
    video_ids: Sequence[str],
    future_head: torch.nn.Module,
) -> tuple[torch.Tensor, int]:
    """Predict next visual window inside the current chunk only.

    Cross-chunk targets are intentionally omitted in this first implementation
    because keeping the next chunk's visual tensor attached to the current
    truncated graph is easy to get wrong.  SSM state is still carried across
    chunks; this auxiliary loss only uses contiguous in-chunk pairs.
    """
    preds = []
    targets = []
    B, W = valid_mask.shape
    for b in range(B):
        _ = video_ids[b]
        for w in range(W - 1):
            if not bool(valid_mask[b, w]) or not bool(valid_mask[b, w + 1]):
                continue
            if int(window_indices[b, w + 1]) != int(window_indices[b, w]) + 1:
                continue
            preds.append(future_head(states[b, w]))
            targets.append(visual_windows[b, w + 1].detach())
    if not preds:
        return states.new_zeros(()), 0
    pred_t = torch.stack(preds, dim=0)
    target_t = torch.stack(targets, dim=0)
    return F.smooth_l1_loss(pred_t, target_t), int(pred_t.shape[0])


def collect_summary_triggers(batch: dict, valid_mask: torch.Tensor) -> tuple[List[tuple[int, int, str]], int]:
    """Return ``(batch_index, window_index, text)`` triggers and skipped count."""
    triggers: List[tuple[int, int, str]] = []
    skipped = 0
    per_batch = batch.get("summary_triggers", [])
    per_skip = batch.get("skipped_summary_boundaries", [])
    for b, window_triggers in enumerate(per_batch):
        if b < len(per_skip):
            skipped += int(sum(per_skip[b]))
        for w, items in enumerate(window_triggers):
            if w >= valid_mask.shape[1] or not bool(valid_mask[b, w]):
                continue
            for item in items:
                text = str(item.get("text", ""))
                if text:
                    triggers.append((b, w, text))
    return triggers, skipped


def build_summary_query_batch(
    embed_fn: torch.nn.Module,
    tokenizer,
    state_tokens: torch.Tensor,
    summary_query: torch.Tensor,
    summary_texts: Sequence[str],
) -> dict:
    """Build ``[state] [summary_query] [caption tokens]`` teacher-forcing batch."""
    device = state_tokens.device
    N, H = state_tokens.shape
    if N != len(summary_texts):
        raise ValueError("state token count must match summary text count")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    token_lists: List[List[int]] = []
    max_len = 0
    for text in summary_texts:
        ids = list(tokenizer.encode(text, add_special_tokens=False)) + [int(eos_id)]
        token_lists.append(ids)
        max_len = max(max_len, len(ids))

    query = summary_query.to(device=device, dtype=state_tokens.dtype).reshape(1, 1, H).expand(N, 1, H)
    base = torch.cat([state_tokens.unsqueeze(1), query], dim=1)
    text_embeds = torch.zeros(N, max_len, H, device=device, dtype=state_tokens.dtype)
    labels = torch.full((N, 2 + max_len), IGNORE_INDEX, dtype=torch.long, device=device)
    for i, ids in enumerate(token_lists):
        if ids:
            id_t = torch.tensor(ids, dtype=torch.long, device=device)
            text_embeds[i, :len(ids)] = embed_fn(id_t)
            labels[i, 2:2 + len(ids)] = id_t
    return {
        "inputs_embeds": torch.cat([base, text_embeds], dim=1),
        "attention_mask": torch.ones(N, 2 + max_len, dtype=torch.bool, device=device),
        "labels": labels,
        "caption_token_count": int(sum(len(ids) for ids in token_lists)),
        "num_triggers": N,
    }


def summary_ce_loss(
    qwen,
    embed_fn: torch.nn.Module,
    tokenizer,
    state_tokens: torch.Tensor,
    summary_query: torch.Tensor,
    summary_texts: Sequence[str],
) -> tuple[torch.Tensor, dict]:
    if len(summary_texts) == 0:
        return state_tokens.new_zeros(()), {
            "num_summary_triggers": 0,
            "caption_token_count": 0,
        }
    batch = build_summary_query_batch(
        embed_fn, tokenizer, state_tokens, summary_query, summary_texts,
    )
    out = qwen(
        inputs_embeds=batch["inputs_embeds"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits
    labels = batch["labels"]
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != IGNORE_INDEX
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(shift_labels.shape)
    per_trigger = (ce * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
    loss = per_trigger.mean()
    return loss, {
        "num_summary_triggers": int(batch["num_triggers"]),
        "caption_token_count": int(batch["caption_token_count"]),
    }


def detach_state_cache(state_cache: dict) -> dict:
    out = {}
    for video_id, state in state_cache.items():
        out[video_id] = {k: v.detach() for k, v in state.items()}
    return out
