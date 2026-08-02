"""Temporal token reduction: removes static patches across video frames.

For each spatial position, compares frames 1..T-1 to frame 0 via cosine similarity.
Patches whose average difference across all non-first frames falls below a threshold
are considered "static" and dropped from all frames except the first.

Original: model/LLM.py temporal_token_reduction()
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTokenReducer(nn.Module):
    """Drop patches that stay nearly identical across frames.

    Args:
        threshold: cosine dissimilarity threshold below which a patch is
                   considered static and dropped. Default 0.01.
        spatial_merge_size: internal spatial grouping factor of the ViT's
                   patch_embed layer. For Qwen2-VL this is 4 (2×2 merge).
                   Default 4.
    """

    def __init__(self, threshold: float = 0.01, spatial_merge_size: int = 4):
        super().__init__()
        self.threshold = threshold
        self.spatial_merge_size = spatial_merge_size

    @torch.no_grad()
    def forward(
        self,
        patch_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute keep/drop mask for patches.

        Args:
            patch_embeds: [total_patches, C]. All patches from all samples in
                         the batch, concatenated along dim 0. C is the ViT
                         patch_embed output dimension (1280 for Qwen2-VL).
            grid_thw: [B, 3]. Each row is (t_frames, h_patches, w_patches)
                      for one sample.

        Returns:
            mask: [total_patches] float tensor. 1.0 = keep, 0.0 = drop.
            frame_token_counts: [sum(t_i)] int tensor. Number of kept patches
                               for every frame across all samples.
        """
        batch_size = grid_thw.shape[0]
        device = patch_embeds.device
        s = self.spatial_merge_size

        token_nums = grid_thw.prod(dim=1)                     # [B]
        split_indices = token_nums.tolist()
        batch_patches = torch.split(patch_embeds, split_indices, dim=0)

        all_masks: List[torch.Tensor] = []
        all_counts: List[torch.Tensor] = []

        for i in range(batch_size):
            t, h, w = grid_thw[i].tolist()
            sample = batch_patches[i]                         # [t*h*w, C]
            # group every s spatially-adjacent patches together
            #  to work at a coarser level: [t, h*w//s, C*s]
            sample_3d = sample.reshape(t, h * w // s, -1)

            if t > 1:
                ref = sample_3d[0:1]                          # [1, h*w//s, C*s]
                ref_expanded = ref.expand(t, h * w // s, -1)  # [t, h*w//s, C*s]
                cos_sim = F.cosine_similarity(sample_3d, ref_expanded, dim=-1)
                diffs = 1.0 - cos_sim                         # [t, h*w//s]

                avg_diff = diffs[1:].mean(dim=0)              # [h*w//s]
                static = avg_diff < self.threshold            # [h*w//s]

                mask_coarse = torch.ones(t, h * w // s, device=device, dtype=torch.bool)
                mask_coarse[1:, static] = False
                mask = torch.repeat_interleave(mask_coarse, s, dim=1)
                mask = mask.reshape(t, h, w)                  # [t, h, w]
            else:
                mask = torch.ones(t, h, w, device=device, dtype=torch.bool)

            all_masks.append(mask.flatten())                  # [t*h*w]
            counts = mask.reshape(t, -1).sum(dim=1).int()     # [t]
            all_counts.append(counts)

        final_mask = torch.cat(all_masks, dim=0)
        final_counts = torch.cat(all_counts, dim=0)
        return final_mask, final_counts
