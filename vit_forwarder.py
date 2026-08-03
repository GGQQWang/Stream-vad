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

from temporal import TemporalTokenReducer


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
            # per-clip keep ratios (each clip may have different H,W)
            ptr = 0
            clip_ratios: List[float] = []
            for row in grid_thw.tolist():
                tg, hg, wg = int(row[0]), int(row[1]), int(row[2])
                n = tg * hg * wg
                clip_ratios.append(mask[ptr: ptr + n].float().mean().item())
                ptr += n
            result = result + ({
                "clip_keep_ratios": clip_ratios,
            },)

        return result

    def forward_batch_micro(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        micro_batch_size: int = 1,
        return_stats: bool = False,
    ):
        """ViT forward in micro-batches to reduce peak attention memory.

        Output is identical to ``forward_batch`` — only the execution
        is split across multiple smaller calls.

        Args:
            pixel_values: ``[total_pixels, C=3]``.
            grid_thw: ``[B, 3]``  rows ``(t_frames, h, w)``.
            micro_batch_size: max clips per micro-batch.  Default 1.
            return_stats: forwarded to ``forward_batch``.

        Returns:
            Same as ``forward_batch``.
        """
        num_clips = int(grid_thw.shape[0])
        if micro_batch_size <= 0 or num_clips <= micro_batch_size:
            return self.forward_batch(pixel_values, grid_thw, return_stats=return_stats)

        patch_counts = (
            grid_thw.detach()
            .to(device="cpu", dtype=torch.long)
            .prod(dim=1)
            .tolist()
        )
        expected = sum(patch_counts)
        actual = int(pixel_values.shape[0])
        if expected != actual:
            raise RuntimeError(
                f"pixel/grid mismatch: expected {expected} patches, got {actual}"
            )

        pixel_clips = torch.split(pixel_values, patch_counts, dim=0)

        token_parts = []
        count_parts = []
        ratio_parts = [] if return_stats else None

        for start in range(0, num_clips, micro_batch_size):
            end = min(start + micro_batch_size, num_clips)
            micro_pixels = torch.cat(pixel_clips[start:end], dim=0)
            micro_grid = grid_thw[start:end]

            fwd = self.forward_batch(
                micro_pixels, micro_grid, return_stats=return_stats,
            )
            if return_stats:
                micro_tokens, micro_counts, micro_stats = fwd
                ratio_parts.extend(micro_stats["clip_keep_ratios"])
            else:
                micro_tokens, micro_counts = fwd

            token_parts.append(micro_tokens)
            count_parts.append(micro_counts)

        result = (torch.cat(token_parts, dim=0), torch.cat(count_parts, dim=0))
        if return_stats:
            result = result + ({"clip_keep_ratios": ratio_parts},)
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
