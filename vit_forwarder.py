"""ViT forward wrapper — the heavy part of the visual pipeline.

Bridges StreamWindowManager / TemporalTokenReducer output with
Qwen2-VL's ViT transformer blocks and merger.  Two modes:

- ``forward_batch()``   training:  pixel_values → patch_embed → temporal
                         compress → ViT blocks → merger.
- ``forward_streaming()`` inference: pre-filtered patches → ViT blocks
                         → merger.

Handles both old ``rotary_pos_emb`` and new ``position_embeddings=(cos,sin)``
ViT-block APIs, and both single-tensor / tuple return from ``rot_pos_emb()``.
"""

import inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal import TemporalTokenReducer


def _detect_vit_api(visual: nn.Module) -> str:
    """Check which keyword the ViT blocks expect."""
    sig = inspect.signature(visual.blocks[0].forward)
    if "position_embeddings" in sig.parameters:
        return "position_embeddings"
    return "rotary_pos_emb"


def _to_position_embeddings(rotary: torch.Tensor):
    """Convert a flat RoPE tensor to ``(cos, sin)`` if needed.

    Some Qwen2-VL versions return ``(cos, sin)`` from rot_pos_emb(),
    others return a single concatenated tensor.  Normalise here.
    """
    if isinstance(rotary, tuple):
        return rotary                     # already (cos, sin)
    # Legacy single-tensor format — split along last dim
    cos, sin = rotary.chunk(2, dim=-1)
    return cos, sin


class ViTForwarder(nn.Module):
    """Qwen2-VL ViT blocks + merger, batch & streaming.

    Args:
        visual: Qwen2-VL ``self.visual`` module (from a loaded model).
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
        self._block_api = _detect_vit_api(visual)  # "rotary_pos_emb" or "position_embeddings"

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
            pixel_values: ``[total_pixels, C=3]``.
            grid_thw: ``[B, 3]``  rows ``(t_frames, h, w)``.

        Returns:
            visual_tokens:  ``[total_tokens, D_llm]``.
            merged_counts:  ``[sum(t_i)]``  tokens / frame post-merger.
        """
        patches = self.visual.patch_embed(pixel_values)          # [total, 1280]
        rotary_raw = self.visual.rot_pos_emb(grid_thw)           # tensor or (cos,sin)

        mask, seqlens = self.temporal_reducer(patches, grid_thw)

        patches = patches[mask.bool()]
        rotary_raw = self._slice_rotary(rotary_raw, mask)

        cu = F.pad(seqlens.cumsum(dim=0), (1, 0), value=0).int()

        patches = self._run_blocks(patches, cu, rotary_raw)

        tokens = self.visual.merger(patches)                     # [L/4, D_llm]
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

        Args:
            all_patches: ``[L, 1280]``  kept patches.
            cu_seqlens:  ``[n_frames + 1]``.
            grid_thw:    ``[n_frames, 3]``.
            all_indices: ``[L]``  patch indices in full-frame RoPE grid.

        Returns:
            visual_tokens: ``[L/4, D_llm]``.
        """
        rotary_full = self.visual.rot_pos_emb(grid_thw)          # [N_full, ...]
        rotary_sel = self._slice_rotary(rotary_full, all_indices)

        all_patches = self._run_blocks(all_patches, cu_seqlens, rotary_sel)
        return self.visual.merger(all_patches)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_blocks(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary: torch.Tensor,
    ) -> torch.Tensor:
        """Run all ViT blocks, adapting to the detected API."""
        if self._block_api == "position_embeddings":
            pos = _to_position_embeddings(rotary)
            for blk in self.visual.blocks:
                x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=pos)
        else:
            for blk in self.visual.blocks:
                x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary)
        return x

    @staticmethod
    def _slice_rotary(
        rotary,
        mask_or_indices: torch.Tensor,
    ):
        """Select positions from a RoPE tensor (handles both formats)."""
        if isinstance(rotary, tuple):
            # (cos, sin) — index each
            idx = mask_or_indices
            if idx.dtype == torch.bool:
                idx = torch.where(idx)[0]
            return (rotary[0][idx], rotary[1][idx])
        # flat tensor
        return rotary[mask_or_indices]
