"""Small tensor helpers for video-level MIL training.

These functions are intentionally independent of Qwen, ViT, and dataset
loading so they can be tested with tiny CPU tensors.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoChunkRef:
    video_id: str
    chunk_start: int
    index: int


def anomaly_logits_from_nll(
    normal_nll: torch.Tensor,
    abnormal_nll: torch.Tensor,
) -> torch.Tensor:
    """Positive means the Abnormal candidate is more likely than Normal."""
    return normal_nll - abnormal_nll


def anomaly_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def mil_ranking_loss(
    abnormal_max_logit: torch.Tensor,
    normal_max_logit: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return F.relu(abnormal_max_logit.new_tensor(margin) - abnormal_max_logit + normal_max_logit)


def select_global_max(chunks: Sequence[Tuple[int, torch.Tensor]]) -> Tuple[int, torch.Tensor]:
    """Return the global window index and value of the max over all chunks.

    Args:
        chunks: sequence of ``(chunk_start, scores)`` where scores contains
            only valid, non-padding windows for that chunk.
    """
    best_index = -1
    best_value = None
    for chunk_start, scores in chunks:
        if scores.numel() == 0:
            continue
        value, local_idx = torch.max(scores, dim=0)
        global_idx = int(chunk_start) + int(local_idx.item())
        if best_value is None or value.item() > best_value.item():
            best_value = value
            best_index = global_idx
    if best_value is None:
        raise ValueError("cannot select max from an empty video bag")
    return best_index, best_value


def normal_language_loss(normal_nll: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    if valid.sum().item() == 0:
        raise ValueError("normal bag has no valid windows")
    return normal_nll[valid].mean()


def abnormal_language_loss(
    abnormal_nll: torch.Tensor,
    selected_global_index: int,
    global_indices: torch.Tensor,
) -> torch.Tensor:
    selected = global_indices == int(selected_global_index)
    if selected.sum().item() != 1:
        raise ValueError("selected abnormal window must appear exactly once")
    return abnormal_nll[selected].squeeze(0)


def abnormal_sparsity_loss(anomaly_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    if valid.sum().item() == 0:
        raise ValueError("abnormal bag has no valid windows")
    return torch.sigmoid(anomaly_logits[valid]).mean()


def group_video_chunks(samples: Sequence[dict]) -> Dict[str, List[VideoChunkRef]]:
    grouped: Dict[str, List[VideoChunkRef]] = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(sample["video_id"], []).append(
            VideoChunkRef(
                video_id=sample["video_id"],
                chunk_start=int(sample["chunk_start"]),
                index=idx,
            )
        )
    for refs in grouped.values():
        refs.sort(key=lambda r: r.chunk_start)
    return grouped


def infer_video_labels(samples: Sequence[dict]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for sample in samples:
        if "video_label" in sample:
            label = int(sample["video_label"])
        else:
            label = 1 if any(int(v) > 0 for v in sample.get("clip_bin", [])) else 0
        labels[sample["video_id"]] = max(labels.get(sample["video_id"], 0), label)
    return labels


def split_normal_abnormal_videos(samples: Sequence[dict]) -> Tuple[List[str], List[str]]:
    labels = infer_video_labels(samples)
    normal = sorted([vid for vid, label in labels.items() if label == 0])
    abnormal = sorted([vid for vid, label in labels.items() if label == 1])
    return normal, abnormal


def count_video_windows(samples: Sequence[dict], video_id: str) -> int:
    total = 0
    for sample in samples:
        if sample["video_id"] == video_id:
            total += int(sample["chunk_end"]) - int(sample["chunk_start"])
    return total


def cycle_pairs(
    normal_videos: Sequence[str],
    abnormal_videos: Sequence[str],
    *,
    generator: torch.Generator | None = None,
) -> List[Tuple[str, str]]:
    """Pair videos, cycling the smaller class to match the larger class."""
    if not normal_videos or not abnormal_videos:
        raise ValueError("MIL training requires at least one normal and one abnormal video")

    n_perm = torch.randperm(len(normal_videos), generator=generator).tolist()
    a_perm = torch.randperm(len(abnormal_videos), generator=generator).tolist()
    normal = [normal_videos[i] for i in n_perm]
    abnormal = [abnormal_videos[i] for i in a_perm]

    n_pairs = max(len(normal), len(abnormal))
    return [(normal[i % len(normal)], abnormal[i % len(abnormal)]) for i in range(n_pairs)]


def finite_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))
