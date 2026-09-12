"""History-conditioned FiLM modulation for spatial visual tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialFiLM(nn.Module):
    """Map SSM internal state to shared gamma/beta for all spatial tokens."""

    def __init__(self, d_ssm: int, llm_hidden: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or max(int(d_ssm) * 2, int(llm_hidden) // 4))
        self.net = nn.Sequential(
            nn.LayerNorm(int(d_ssm)),
            nn.Linear(int(d_ssm), hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * int(llm_hidden)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        h_internal: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must be [N, R, H], got {tuple(spatial_tokens.shape)}")
        if h_internal.ndim != 2:
            raise ValueError(f"h_internal must be [N, d_ssm], got {tuple(h_internal.shape)}")
        if spatial_tokens.shape[0] != h_internal.shape[0]:
            raise ValueError(
                f"batch mismatch: spatial_tokens={tuple(spatial_tokens.shape)}, "
                f"h_internal={tuple(h_internal.shape)}"
            )
        param = next(self.parameters())
        x = spatial_tokens.to(device=param.device, dtype=param.dtype)
        h = h_internal.to(device=param.device, dtype=param.dtype)
        gamma_raw, beta = self.net(h).chunk(2, dim=-1)
        gamma = torch.tanh(gamma_raw)
        modulated = (1.0 + gamma.unsqueeze(-2)) * x + beta.unsqueeze(-2)
        with torch.no_grad():
            diff = (modulated - x).detach().float()
            base = x.detach().float()
            rel = diff.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-8)
            stats = {
                "film_gamma_abs_mean": float(gamma.detach().float().abs().mean().item()),
                "film_beta_abs_mean": float(beta.detach().float().abs().mean().item()),
                "film_delta_rel": float(rel.mean().item()),
            }
        return modulated, stats
