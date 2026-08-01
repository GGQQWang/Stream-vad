"""ViT forward wrapper — the heavy part of the visual pipeline.

Bridges StreamWindowManager / TemporalTokenReducer output with
Qwen2-VL's ViT transformer blocks and merger.  Two modes:

- ``forward_batch()``   training:  pixel_values → patch_embed → temporal
                         compress → ViT blocks → merger.
- ``forward_streaming()`` inference: pre-filtered patches → ViT blocks
                         → merger.

Does NOT import from LAVIDA.  References Qwen2-VL modules directly.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.temporal import TemporalTokenReducer


class ViTForwarder(nn.Module):
    """Qwen2-VL ViT blocks + merger, batch & streaming.

    Args:
        visual: Qwen2-VL ``self.visual`` module (from a loaded
                ``Qwen2VLForConditionalGeneration`` model).
        temporal_reducer: TemporalTokenReducer instance for batch mode.
                         If None, batch mode still works (no temporal
                         filtering).
    """

    def __init__(
        self,
        visual: nn.Module,
        temporal_reducer: Optional[TemporalTokenReducer] = None,
    ):
        super().__init__()
        self.visual = visual
        self.temporal_reducer = temporal_reducer or TemporalTokenReducer()

    # ------------------------------------------------------------------
    # Batch mode (training: uniform clip sampling)
    # ------------------------------------------------------------------
    def forward_batch(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of video clips.

        Args:
            pixel_values: ``[total_pixels, C=3]``  raw pixel values,
                         flattened across all clips in the batch.
            grid_thw: ``[B, 3]``  rows are ``(t_frames, h, w)``.

        Returns:
            visual_tokens:  ``[total_tokens, D_llm]``  merger output
                           (``D_llm`` = 3584 for Qwen2-VL-7B).
            merged_counts:  ``[sum(t_i)]``  token count per frame
                           **after merger** (original / 4).
        """
        # 1. patch_embed  (same as LAVIDA video_embed L365)
        patches = self.visual.patch_embed(pixel_values)          # [total, 1280]
        rotary = self.visual.rot_pos_emb(grid_thw)               # [total, 1280]

        # 2. temporal filtering  (our module replaces LAVIDA L368)
        mask, seqlens = self.temporal_reducer(patches, grid_thw)

        patches = patches[mask.bool()]                           # [kept, 1280]
        rotary = rotary[mask.bool()]

        # 3. cu_seqlens for Flash Attention varlen  (same as LAVIDA L370-371)
        cu = F.pad(seqlens.cumsum(dim=0), (1, 0), value=0).int()

        # 4. ViT blocks  (same as LAVIDA L374-375)
        for blk in self.visual.blocks:
            patches = blk(patches, cu_seqlens=cu, rotary_pos_emb=rotary)

        # 5. merger  (same as LAVIDA L414)
        tokens = self.visual.merger(patches)                     # [kept/4, D_llm]

        # per-frame count after 4× downsample
        merged_counts = torch.ceil(seqlens.float() / 4).int()

        return tokens, merged_counts

    # ------------------------------------------------------------------
    # Streaming mode (inference: adaptive flush)
    # ------------------------------------------------------------------
    def forward_streaming(
        self,
        all_patches: torch.Tensor,
        cu_seqlens: torch.Tensor,
        grid_thw: torch.Tensor,
        all_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Process one flushed window.

        Patches are already filtered by ``StreamWindowManager``.
        We only need to compute matching position embeddings and run
        ViT blocks + merger.

        Args:
            all_patches: ``[L, 1280]``  kept patches from the window.
            cu_seqlens:  ``[n_frames + 1]``  cumulative lengths.
            grid_thw:    ``[n_frames, 3]``  rows ``(1, h, w)``.
            all_indices: ``[L]``  per-patch index in the original
                        (full-frame) position embedding grid, used
                        to select the right rotary positions.

        Returns:
            visual_tokens: ``[L/4, D_llm]``  merger output.
        """
        # 1. full rotary embeddings for every frame, then select kept ones
        rotary_full = self.visual.rot_pos_emb(grid_thw)          # [total_full, 1280]
        rotary = rotary_full[all_indices]                         # [L, 1280]

        # 2. ViT blocks
        for blk in self.visual.blocks:
            all_patches = blk(
                all_patches,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary,
            )

        # 3. merger
        return self.visual.merger(all_patches)                   # [L/4, D_llm]
