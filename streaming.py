"""Streaming window manager for adaptive frame gating.

Manages frame-level flush decisions and patch-level temporal compression.
Each incoming frame is compared against a reference; frames with sufficient
change are accumulated. When enough changed frames collect (or a timeout
fires), the buffer is flushed for downstream ViT+LLM processing.

After flush, the reference resets to the last kept frame, making each
window self-contained with no long-term state drift.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamWindowManager(nn.Module):
    """Adaptive frame gating and patch buffer for streaming VAD.

    Frame-level (flush):
        Compares each frame to the window reference via cosine similarity.
        If the fraction of changed patches exceeds `patch_change_ratio`,
        the frame is counted as "interesting".

    Patch-level (temporal compression):
        Within each kept frame, only patches whose cosine diff exceeds
        `change_threshold` are retained. Static patches are dropped.

    Args:
        min_changed_frames: number of interesting frames needed to flush.
                           Default 8 (matches LAVIDA total_sampled_frames).
        max_wait_frames: force-flush after this many total frames regardless
                        of content. Prevents indefinite waiting on static
                        scenes. Default 300 (~10s at 30fps).
        change_threshold: cosine dissimilarity below which a patch is
                         considered static and dropped. Default 0.01.
        patch_change_ratio: fraction of patches that must change for the
                           whole frame to be deemed interesting. Default 0.15.
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
        self._ref: Optional[torch.Tensor] = None     # [N_patches, C]
        self._buffer: List[torch.Tensor] = []         # kept-patch tensors per frame
        self._counts: List[int] = []                  # kept-patch count per frame
        self._changed: int = 0                        # interesting-frame count
        self._total: int = 0                          # total frames seen
        self._last_kept: Optional[torch.Tensor] = None  # last kept frame (all patches)

    # ------------------------------------------------------------------
    # Per-frame entry point
    # ------------------------------------------------------------------
    @torch.no_grad()
    def add_frame(
        self,
        patch_embeds: torch.Tensor,
    ) -> bool:
        """Process one frame.  Returns True if the frame was kept.

        Args:
            patch_embeds: [N, C]  ViT patch_embed output for a single frame.
                          N = h * w (or h*w//4 with Qwen2-VL internal merge).

        Returns:
            True  if this frame had enough change to be counted.
            False if it was mostly static and skipped.
        """
        # --- first frame of a new window: always keep, set as reference ---
        if self._ref is None:
            self._ref = patch_embeds.clone()
            self._buffer.append(patch_embeds)
            self._counts.append(patch_embeds.shape[0])
            self._changed = 1
            self._total = 1
            self._last_kept = patch_embeds.clone()
            return True

        self._total += 1

        # --- compare every patch position to reference ---
        diff = 1.0 - F.cosine_similarity(patch_embeds, self._ref, dim=-1)  # [N]
        changed_mask = diff > self.threshold                              # [N] bool
        ratio = changed_mask.float().mean().item()

        if ratio >= self.patch_ratio:
            kept = patch_embeds[changed_mask]             # [n_kept, C]
            self._buffer.append(kept)
            self._counts.append(kept.shape[0])
            self._changed += 1
            self._last_kept = patch_embeds.clone()
            return True

        # frame mostly static — skip entirely
        return False

    # ------------------------------------------------------------------
    # Flush decision
    # ------------------------------------------------------------------
    def should_flush(self) -> bool:
        """Check whether the current window is ready to be flushed."""
        return (
            self._changed >= self.min_changed
            or self._total >= self.max_wait
        )

    @property
    def changed_frames(self) -> int:
        return self._changed

    @property
    def total_frames(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Flush: produce packed sequence for ViT blocks
    # ------------------------------------------------------------------
    def flush(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack accumulated patches and reset for the next window.

        Returns:
            all_patches: [total_kept, C]  concatenated patches across all
                        kept frames, ready for ViT.blocks().
            cu_seqlens:  [n_frames + 1]  int32 cumulative sequence lengths
                        for Flash Attention varlen.
        """
        if not self._buffer:
            raise RuntimeError("flush() called with empty buffer")

        all_patches = torch.cat(self._buffer, dim=0)       # [L, C]
        counts = torch.tensor(self._counts, dtype=torch.int32)
        cu_seqlens = F.pad(counts.cumsum(dim=0), (1, 0), value=0).int()

        # ---- reset for next window ----
        # Clear reference so the next incoming frame automatically becomes
        # the new window's first frame (unconditionally kept, no comparison).
        # Each window is self-contained — no state drift between windows.
        self._ref = None
        self._buffer = []
        self._counts = []
        self._changed = 0
        self._total = 0
        self._last_kept = None

        return all_patches, cu_seqlens
