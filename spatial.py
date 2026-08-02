"""Spatial token compression via LOF-based clustering.

Reduces the number of visual tokens by:
1. Estimating local density of each token via Local Outlier Factor (LOF)
2. Selecting high-density tokens as cluster centres
3. Merging tokens within each cluster via attention-weighted averaging

This preserves distinctive (low-density) information while collapsing
redundant (high-density) regions.

Original: model/LLM.py visual_token_sampling()
"""

from typing import Tuple

import torch
import torch.nn as nn


class SpatialTokenCompressor(nn.Module):
    """Merge redundant visual tokens via LOF clustering + attention.

    Args:
        reduction_ratio: fraction of tokens to keep after compression.
                       0.5 means keep 50%. Default 0.5 (from LAVIDA config).
        k: number of nearest neighbours for LOF density estimation.
           Default 8 (from LAVIDA config).
    """

    def __init__(self, reduction_ratio: float = 0.5, k: int = 8):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.k = k

    @torch.no_grad()
    def forward(
        self,
        visual_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress visual token sequence.

        Args:
            visual_embeds: [L, C]. Stack of visual token embeddings.

        Returns:
            merged_features: [R, C]. Compressed token embeddings,
                            R = max(1, ceil(L * reduction_ratio)).
            kept_indices: [R] long tensor. Indices of selected cluster centres
                         in the original sequence (sorted ascending).
        """
        L, C = visual_embeds.shape
        k = min(self.k, L - 1)  # can't have more neighbours than tokens
        if k < 1:
            return visual_embeds, torch.arange(L, device=visual_embeds.device)

        # work in float32 for numerical stability
        embeds = visual_embeds.to(torch.float32)
        device = embeds.device
        scale = C ** 0.5

        # ---- LOF density estimation ----
        dist = torch.cdist(embeds, embeds) / scale             # [L, L]
        # nearest[0] is self (distance 0), take k neighbours beyond self
        d_nearest, idx_nearest = torch.topk(dist, k=k + 1, dim=-1, largest=False)
        k_dists = d_nearest[:, 1:]                              # [L, k]
        neighbours = idx_nearest[:, 1:]                          # [L, k]

        # reachability distance
        k_dist_vals = k_dists[:, -1]                            # [L]
        k_dist_expanded = k_dist_vals.unsqueeze(0).expand(L, L) # [L, L]
        reach = torch.maximum(k_dist_expanded, dist)            # [L, L]

        # gather reachability distance to each point's k neighbours
        reach_nbr = torch.gather(reach, dim=1, index=neighbours)# [L, k]

        # local reachability density  LRD(p) = k / Σ reach_dist(p, o)
        lrd = k / (reach_nbr.sum(dim=1) + 1e-10)                # [L]

        # ---- select cluster centres ----
        R = max(1, int(L * self.reduction_ratio))
        _, topk_idx = torch.topk(lrd, k=R, dim=0, largest=True)
        centres_idx, _ = torch.sort(topk_idx)                   # [R]
        centres = embeds[centres_idx]                           # [R, C]

        # ---- hard assignment + soft merge ----
        d2c = torch.cdist(embeds, centres) / scale              # [L, R]
        assignments = torch.argmin(d2c, dim=1)                  # [L]

        one_hot = torch.zeros(L, R, device=device)
        one_hot.scatter_(1, assignments.unsqueeze(1), 1.0)      # [L, R]
        cluster_mask = one_hot.t().bool()                       # [R, L]

        # attention within each cluster
        attn = centres @ embeds.t() / scale                     # [R, L]
        attn = -attn                                            # negate: dot → distance proxy
        attn = attn.masked_fill(~cluster_mask, -1e9)
        weights = torch.softmax(attn, dim=1)                    # [R, L]
        merged = weights @ embeds                               # [R, C]

        return merged.to(visual_embeds.dtype), centres_idx
