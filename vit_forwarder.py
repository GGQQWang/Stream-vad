"""ViT forward wrapper — the heavy part of the visual pipeline.

Bridges StreamWindowManager / TemporalTokenReducer output with
Qwen2-VL's ViT transformer blocks and merger.  Two modes:

- ``forward_batch()``   training:  pixel_values → patch_embed → temporal
                         compress → ViT blocks → merger.
- ``forward_streaming()`` inference: pre-filtered patches → ViT blocks
                         → merger.

Tested with transformers==4.46.2 (``rotary_pos_emb`` API).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal import TemporalTokenReducer


class ViTForwarder(nn.Module):
    """Qwen2-VL ViT blocks + merger, batch & streaming.

    Args:
        visual: Qwen2-VL ``self.visual`` module.
        temporal_reducer: TemporalTokenReducer for batch mode.
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
        return_stats: bool = False,
    ):
        """Process a batch of video clips.

        Args:
            pixel_values: ``[total_pixels, C=3]``.
            grid_thw: ``[B, 3]``  rows ``(t_frames, h, w)``.
            return_stats: if True, also return a compression-stats dict.

        Returns:
            visual_tokens:  ``[total_tokens, D_llm]``.
            merged_counts:  ``[sum(t_i)]``  tokens / frame post-merger.
            stats (only if return_stats): dict with per-clip keep_ratios
                  and per-clip anomaly-friendly breakdown.
        """
        patches = self.visual.patch_embed(pixel_values)          # [total, 1280]
        rotary = self.visual.rot_pos_emb(grid_thw)               # [total, 1280]

        mask, seqlens = self.temporal_reducer(patches, grid_thw)

        patches = patches[mask]
        rotary = rotary[mask]

        cu = F.pad(seqlens.cumsum(dim=0), (1, 0), value=0).int()

        for blk in self.visual.blocks:
            patches = blk(patches, cu_seqlens=cu, rotary_pos_emb=rotary)

        tokens = self.visual.merger(patches)                     # [L/4, D_llm]
        merged_counts = torch.ceil(seqlens.float() / 4).int()

        result = (tokens, merged_counts)

        if return_stats:
            # per-clip keep ratios
            tg_list = grid_thw[:, 0].tolist()                    # list[int]
            ptr = 0
            clip_ratios: List[float] = []
            for tg in tg_list:
                n = tg * grid_thw[0, 1].item() * grid_thw[0, 2].item()
                clip_ratios.append(mask[ptr: ptr + n].float().mean().item())
                ptr += n
            result = result + ({
                "keep_mask": mask.cpu(),
                "clip_keep_ratios": clip_ratios,
            },)

        return result

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

        Args:
            all_patches: ``[L, 1280]``  kept patches.
            cu_seqlens:  ``[n_frames + 1]``.
            grid_thw:    ``[n_frames, 3]``.
            all_indices: ``[L]``  patch indices in full-frame RoPE grid.

        Returns:
            visual_tokens: ``[L/4, D_llm]``.
        """
        rotary_full = self.visual.rot_pos_emb(grid_thw)          # [N_full, 1280]
        rotary = rotary_full[all_indices]                         # [L, 1280]

        for blk in self.visual.blocks:
            all_patches = blk(
                all_patches, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary,
            )
        return self.visual.merger(all_patches)
