"""Streaming window manager for adaptive frame gating.

Manages frame-level flush decisions and patch-level temporal compression.
Each incoming frame is compared against a reference; frames with sufficient
change are accumulated.  When enough changed frames collect (or a timeout
fires), the buffer is flushed for downstream ViT+LLM processing.

Patch filtering operates on **2×2 merge groups**, not individual patches,
so the Qwen2-VL merger always receives spatially coherent groups of 4.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamWindowManager(nn.Module):
    """Adaptive frame gating and patch buffer for streaming VAD.

    Frame-level (flush):
        Compares each frame to the window reference via cosine similarity
        **at the 2×2 merge-group level**.  If the fraction of changed
        groups exceeds ``patch_change_ratio``, the frame is "interesting".

    Patch-level (temporal compression):
        Only groups whose cosine diff exceeds ``change_threshold`` are
        retained.  Because groups are kept / dropped as a whole, the patch
        count per frame is always a multiple of 4 — the Qwen2-VL merger
        requirement.

    Args:
        min_changed_frames: interesting frames needed to flush.
                           Default 8 (matches LAVIDA total_sampled_frames).
        max_wait_frames: force-flush after this many total frames.
                        Default 300 (~10 s at 30 fps).
        change_threshold: cosine dissimilarity below which a **group**
                         is considered static.  Default 0.01.
        patch_change_ratio: fraction of **groups** that must change for
                           the frame to be deemed interesting.  Default 0.15.
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
            patch_embeds: ``[N, C]``  ViT patch_embed output.
                          ``N = h * w``.
            grid_hw: ``(h, w)``  spatial grid.

        Returns:
            ``True`` if this frame had enough change to be counted.
        """
        N, C = patch_embeds.shape
        h, w = grid_hw
        assert h * w == N, f"grid {h}×{w}={h*w} ≠ N={N}"
        assert N % 4 == 0, f"patch count {N} not divisible by 4"

        # --- first frame of a new window: keep all, set reference ---
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

        # --- group-level comparison (same pattern as LAVIDA L430) ---
        # Reshape  [h*w, C]  →  [h*w//4, 4*C]
        # Each row is one 2×2 merge group.
        cur_groups = patch_embeds.reshape(N // 4, C * 4)
        ref_groups = self._ref.reshape(N // 4, C * 4)

        group_diff = 1.0 - F.cosine_similarity(
            cur_groups, ref_groups, dim=-1
        )                                                   # [N//4]
        changed_mask = group_diff > self.threshold          # [N//4]

        ratio = changed_mask.float().mean().item()

        if ratio < self.patch_ratio:
            return False                                     # frame skipped

        # --- expand group mask to patch level ---
        # repeat_interleave matches LAVIDA L444
        patch_mask = torch.repeat_interleave(changed_mask, 4)  # [N]

        kept = patch_embeds[patch_mask]                       # [n_kept, C]
        idx = torch.where(patch_mask)[0]                      # [n_kept]

        assert kept.shape[0] % 4 == 0, (
            f"kept patches {kept.shape[0]} not divisible by 4"
        )

        self._buf.append(kept)
        self._counts.append(kept.shape[0])
        self._grids.append(grid_hw)
        self._indices.append(idx)
        self._changed += 1
        return True

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
    # Flush
    # ------------------------------------------------------------------
    def flush(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack accumulated patches and reset.

        Returns:
            all_patches: ``[L, C]``  concatenated kept patches.
                         ``L`` is always a multiple of 4 per frame.
            cu_seqlens:  ``[n_frames + 1]``  int32 cumulative lengths.
            grid_thw:    ``[n_frames, 3]``  rows ``(1, h, w)``.
            all_indices: ``[L]``  per-patch index in the full frame.
        """
        if not self._buf:
            raise RuntimeError("flush() called with empty buffer")

        all_patches = torch.cat(self._buf, dim=0)
        device = all_patches.device

        counts = torch.tensor(self._counts, dtype=torch.int32, device=device)
        cu_seqlens = F.pad(counts.cumsum(dim=0), (1, 0), value=0).int()

        grid_thw = torch.tensor(
            [[1, h, w] for (h, w) in self._grids],
            dtype=torch.int32,
            device=device,
        )
        all_indices = torch.cat(self._indices, dim=0)

        # ---- reset for next window ----
        self._ref = None
        self._buf = []
        self._counts = []
        self._grids = []
        self._indices = []
        self._changed = 0
        self._total = 0

        return all_patches, cu_seqlens, grid_thw, all_indices
