"""Streaming window manager for adaptive frame gating.

Manages frame-level flush decisions and patch-level temporal compression.
Each incoming frame is compared against a reference; frames with sufficient
change are accumulated. When enough changed frames collect (or a timeout
fires), the buffer is flushed for downstream ViT+LLM processing.

After flush, the reference resets, making each window self-contained with
no long-term state drift.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamWindowManager(nn.Module):
    """Adaptive frame gating and patch buffer for streaming VAD.

    Frame-level (flush):
        Compares each frame to the window reference via cosine similarity.
        If the fraction of changed patches exceeds ``patch_change_ratio``,
        the frame is counted as "interesting".

    Patch-level (temporal compression):
        Within each kept frame, only patches whose cosine diff exceeds
        ``change_threshold`` are retained.  Static patches are dropped.
        Per-patch indices are stored so downstream ViT blocks can select
        the matching rotary position embeddings.

    Args:
        min_changed_frames: interesting frames needed to flush.
                           Default 8 (matches LAVIDA total_sampled_frames).
        max_wait_frames: force-flush after this many total frames.
                        Default 300 (~10 s at 30 fps).
        change_threshold: cosine dissimilarity below which a patch is
                         considered static.  Default 0.01.
        patch_change_ratio: fraction of patches that must change for the
                           whole frame to be deemed interesting.  Default 0.15.
    """

    def __init__(
        self,
        min_changed_frames: int = 8,
        max_wait_frames: int = 300,
        change_threshold: float = 0.01,
        patch_change_ratio: float = 0.15,
    ):
        super().__init__()
        self.min_changed = min_changed_frames
        self.max_wait = max_wait_frames
        self.threshold = change_threshold
        self.patch_ratio = patch_change_ratio
        self.reset()

    def reset(self) -> None:
        """Clear all internal state for a new video."""
        self._ref: Optional[torch.Tensor] = None       # [N, C] reference patches
        self._buf: List[torch.Tensor] = []              # kept-patch tensors
        self._counts: List[int] = []                    # kept-patch count / frame
        self._grids: List[Tuple[int, int]] = []         # (h, w) per kept frame
        self._indices: List[torch.Tensor] = []          # patch indices / frame
        self._changed: int = 0
        self._total: int = 0

    # ------------------------------------------------------------------
    # Per-frame entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def add_frame(
        self,
        patch_embeds: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> bool:
        """Process one frame.  Returns ``True`` if the frame was kept.

        Args:
            patch_embeds: ``[N, C]``  ViT patch_embed output for a single
                          frame.  ``N = h * w`` (or ``h*w//4`` with the
                          Qwen2-VL internal 2×2 merge).
            grid_hw: ``(h, w)`` spatial grid of this frame **after**
                     patch_embed.  Needed to construct ``rotary_pos_emb``
                     later in ViTForwarder.

        Returns:
            ``True``  if this frame had enough change to be counted.
            ``False`` if it was mostly static and skipped.
        """
        N = patch_embeds.shape[0]

        # --- first frame of a new window: always keep, set as reference ---
        if self._ref is None:
            self._ref = patch_embeds.clone()
            self._buf.append(patch_embeds)
            self._counts.append(N)
            self._grids.append(grid_hw)
            self._indices.append(torch.arange(N, device=patch_embeds.device))
            self._changed = 1
            self._total = 1
            return True

        self._total += 1

        # --- compare every patch position to reference ---
        diff = 1.0 - F.cosine_similarity(patch_embeds, self._ref, dim=-1)   # [N]
        changed_mask = diff > self.threshold                                # [N]
        ratio = changed_mask.float().mean().item()

        if ratio >= self.patch_ratio:
            idx = torch.where(changed_mask)[0]         # kept patch indices
            self._buf.append(patch_embeds[idx])
            self._counts.append(idx.shape[0])
            self._grids.append(grid_hw)
            self._indices.append(idx)
            self._changed += 1
            return True

        return False

    # ------------------------------------------------------------------
    # Flush decision
    # ------------------------------------------------------------------
    def should_flush(self) -> bool:
        return self._changed >= self.min_changed or self._total >= self.max_wait

    @property
    def changed_frames(self) -> int:
        return self._changed

    @property
    def total_frames(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Flush: produce packed sequence for ViT blocks
    # ------------------------------------------------------------------
    def flush(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack accumulated patches and reset for the next window.

        Returns:
            all_patches: ``[L, C]``  concatenated kept patches.
            cu_seqlens:  ``[n_frames + 1]``  int32 cumulative lengths
                        for Flash Attention varlen.
            grid_thw:    ``[n_frames, 3]``  rows are ``(1, h, w)``,
                        used by ``rotary_pos_emb``.
            all_indices: ``[L]``  per-patch index in the **full** frame,
                        used to select matching rotary position embeddings.
        """
        if not self._buf:
            raise RuntimeError("flush() called with empty buffer")

        all_patches = torch.cat(self._buf, dim=0)
        counts = torch.tensor(self._counts, dtype=torch.int32)
        cu_seqlens = F.pad(counts.cumsum(dim=0), (1, 0), value=0).int()

        grid_thw = torch.tensor(
            [[1, h, w] for (h, w) in self._grids],
            dtype=torch.int32,
        )
        all_indices = torch.cat(self._indices, dim=0)      # [L]

        # ---- reset for next window ----
        self._ref = None
        self._buf = []
        self._counts = []
        self._grids = []
        self._indices = []
        self._changed = 0
        self._total = 0

        return all_patches, cu_seqlens, grid_thw, all_indices
