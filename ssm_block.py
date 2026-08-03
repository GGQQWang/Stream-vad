"""SSM temporal modelling block using Mamba2.

Supports three modes:
- forward(x)              training: differentiable scan over [B,T,d]
- forward_chunk(x,state)  training: chunked forward with state carry + grad
- forward_step(x,seq_idx) inference: single-window, cache-driven

Based on: StreamMind/streammind/model/mamba_ssm/
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    Mamba2 = None
    mamba_chunk_scan_combined = None

from einops import rearrange


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class SSMState:
    """Per-block Mamba2 state for cross-chunk carry.

    conv_state: [B, conv_dim, d_conv]  causal-conv history
    ssm_state:  [B, nheads, headdim, d_state]  SSM hidden state
    """
    conv_state: torch.Tensor
    ssm_state: torch.Tensor

    def detach(self) -> "SSMState":
        return SSMState(
            conv_state=self.conv_state.detach(),
            ssm_state=self.ssm_state.detach(),
        )

    def clone(self) -> "SSMState":
        return SSMState(
            conv_state=self.conv_state.clone(),
            ssm_state=self.ssm_state.clone(),
        )

    def to_device(self, device: torch.device) -> "SSMState":
        return SSMState(
            conv_state=self.conv_state.to(device),
            ssm_state=self.ssm_state.to(device),
        )


# ---------------------------------------------------------------------------
# helpers — extract Mamba2 internals
# ---------------------------------------------------------------------------

def _mamba_chunk_forward(
    blk: nn.Module,
    u: torch.Tensor,                  # [B, L, d_model]
    conv_state: Optional[torch.Tensor] = None,   # [B, D_conv, d_conv]
    ssm_state: Optional[torch.Tensor] = None,    # [B, nheads, headdim, d_state]
) -> Tuple[torch.Tensor, SSMState]:
    """Differentiable chunk forward through one Mamba2 block.

    Returns ``(hidden_states, new_state)``.
    """
    assert not blk.use_mem_eff_path, "forward_chunk requires use_mem_eff_path=False"
    batch, seqlen, _ = u.shape

    # 1. project
    zxbcdt = blk.in_proj(u)                                      # [B, L, d_in_proj]
    d_mlp = (zxbcdt.shape[-1] - 2 * blk.d_ssm - 2 * blk.ngroups * blk.d_state - blk.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt,
        [d_mlp, d_mlp, blk.d_ssm, blk.d_ssm + 2 * blk.ngroups * blk.d_state, blk.nheads],
        dim=-1,
    )

    # 2. causal conv1d with state carry
    xBC_t = rearrange(xBC, "b l d -> b d l")                     # [B, conv_dim, L]

    hist = (
        conv_state[..., -(blk.d_conv - 1):]
        if conv_state is not None
        else xBC_t.new_zeros(batch, xBC_t.shape[1], blk.d_conv - 1)
    )                                                              # [B, conv_dim, d_conv-1]

    if causal_conv1d_fn is not None:
        xBC_conv = causal_conv1d_fn(
            xBC_t,
            rearrange(blk.conv1d.weight, "d 1 w -> d w"),
            blk.conv1d.bias,
            initial_states=hist,
            activation="silu",
        )                                                          # [B, conv_dim, L]
    else:
        xBC_padded = torch.cat([hist, xBC_t], dim=-1)             # [B, conv_dim, L+d_conv-1]
        xBC_conv = F.conv1d(
            xBC_padded, blk.conv1d.weight, blk.conv1d.bias,
            padding=0, groups=blk.conv1d.groups,
        )
        xBC_conv = blk.act(xBC_conv)                              # [B, conv_dim, L]

    xBC = rearrange(xBC_conv, "b d l -> b l d")                  # [B, L, conv_dim]

    # store new conv state: last d_conv inputs of history+current
    state_input = torch.cat([hist, xBC_t], dim=-1)               # [B, conv_dim, L+d_conv-1]
    new_conv_state = state_input[..., -blk.d_conv:]              # [B, conv_dim, d_conv]

    # 3. SSM scan
    dt = F.softplus(dt + blk.dt_bias)
    A = -torch.exp(blk.A_log.float())
    x, B, C = torch.split(
        xBC,
        [blk.d_ssm, blk.ngroups * blk.d_state, blk.ngroups * blk.d_state],
        dim=-1,
    )

    y, final_ssm_state = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=blk.headdim),
        dt,
        A,
        rearrange(B, "b l (g n) -> b l g n", g=blk.ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=blk.ngroups),
        chunk_size=blk.chunk_size,
        D=rearrange(blk.D, "(h p) -> h p", p=blk.headdim) if blk.D_has_hdim else blk.D,
        z=None,
        initial_states=ssm_state,
        return_final_states=True,
    )                                                             # y:[B,L,H,P]  fss:[B,H,P,N]

    y = rearrange(y, "b l h p -> b l (h p)")

    # 4. gate + merge
    y = blk.norm(y, z)
    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    out = blk.out_proj(y)

    new_state = SSMState(
        conv_state=new_conv_state,
        ssm_state=final_ssm_state,
    )
    return out, new_state


# ---------------------------------------------------------------------------
# SSMBlock (main class)
# ---------------------------------------------------------------------------

class SSMBlock(nn.Module):
    """Mamba2 window-to-window temporal model.

    Args:
        d_input: pooled window vector dim.
        d_model: Mamba2 internal dim.  Default 256.
        n_layers: stacked Mamba2 blocks.  Default 1.
        llm_hidden: output dim (LLM hidden_size).  Default 3584.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        n_layers: int = 1,
        llm_hidden: int = 3584,
    ):
        super().__init__()
        if Mamba2 is None:
            raise ImportError("mamba_ssm not installed.")

        self.d_model = d_model
        self.n_layers = n_layers

        self.in_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
        )

        # use_mem_eff_path=False — required for forward_chunk() state support
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2,
                   layer_idx=i, use_mem_eff_path=False)
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, llm_hidden)

        # inference cache (for forward_step streaming)
        self._cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

    # ------------------------------------------------------------------
    # Full-sequence forward (training, no state carry)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full differentiable scan.  ``x``: ``[B, T, d_input]``."""
        h = self.in_proj(x)
        for blk, norm in zip(self.blocks, self.norms):
            residual = h
            h = norm(h)
            h = blk(h)
            h = h + residual
        h = self.out_norm(h)
        return self.out_proj(h)

    # ------------------------------------------------------------------
    # Chunked forward (training with state carry, differentiable)
    # ------------------------------------------------------------------
    def forward_chunk(
        self,
        x: torch.Tensor,
        state: Optional[Dict[int, SSMState]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, SSMState]]:
        """Differentiable chunk forward with SSM state carry.

        Args:
            x: ``[B, T, d_input]``  — one chunk of window vectors.
            state: ``{layer_idx: SSMState}``  from previous chunk,
                  or ``None`` for the first chunk of a video.

        Returns:
            output: ``[B, T, llm_hidden]``
            new_state: ``{layer_idx: SSMState}``  for next chunk.
        """
        h = self.in_proj(x)                                        # [B, T, d_model]

        new_state: Dict[int, SSMState] = {}
        for i, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
            residual = h
            h = norm(h)

            prev = state.get(i) if state else None
            conv_s = prev.conv_state if prev else None
            ssm_s = prev.ssm_state if prev else None

            h, ns = _mamba_chunk_forward(blk, h, conv_state=conv_s, ssm_state=ssm_s)
            new_state[i] = ns
            h = h + residual

        h = self.out_norm(h)
        return self.out_proj(h), new_state

    # ------------------------------------------------------------------
    # Streaming inference (step-by-step, cache-driven, no grad)
    # ------------------------------------------------------------------
    def allocate_cache(self, batch_size: int, max_windows: int) -> None:
        self._cache = {}
        for i, blk in enumerate(self.blocks):
            conv_state, ssm_state = blk.allocate_inference_cache(batch_size, max_windows)
            self._cache[i] = (conv_state, ssm_state)

    @torch.no_grad()
    def forward_step(self, x: torch.Tensor, seq_idx: int) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("allocate_cache() first")
        h = self.in_proj(x)
        for i, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
            residual = h
            h = norm(h)
            conv_state, ssm_state = self._cache[i]
            h, conv_state, ssm_state = blk.step(h, conv_state, ssm_state)
            self._cache[i] = (conv_state, ssm_state)
            h = h + residual
        h = self.out_norm(h)
        return self.out_proj(h)

    def reset_cache(self) -> None:
        self._cache = None

    def get_cache(self, detach: bool = False):
        if self._cache is None:
            return None
        out = {}
        for i, (cs, ss) in self._cache.items():
            out[i] = SSMState(
                conv_state=cs.detach().clone() if detach else cs.clone(),
                ssm_state=ss.detach().clone() if detach else ss.clone(),
            )
        return out

    def set_cache(self, cache: dict | None) -> None:
        if cache is None:
            self._cache = None
            return
        dev = next(self.blocks[0].parameters()).device
        self._cache = {}
        for i, st in cache.items():
            self._cache[i] = (st.conv_state.to(dev), st.ssm_state.to(dev))
