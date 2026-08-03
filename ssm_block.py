"""SSM temporal modelling block using Mamba2.

Wraps mamba_ssm Mamba2 directly with explicit residual management.
Each window produces one pooled token, so Mamba2 operates on
[B, T, D] where T = number of flushed windows per video.

Training:    process all T windows as a sequence.
Streaming:   use inference_params + pre-allocated cache for true
             step-by-step inference (one window at a time, internal
             conv_state + ssm_state carried forward correctly).

Based on: StreamMind/streammind/model/multimodal_projector/ssm.py
          StreamMind/streammind/model/mamba_ssm/modules/mamba2.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    Mamba2 = None


class SSMBlock(nn.Module):
    """Mamba2 window-to-window temporal model.

    Training:
        forward(x)  --  x: [B, T, d_input] -> ssm_tokens: [B, T, llm_hidden]

    Streaming inference:
        1. allocate_cache(B, max_T)
        2. for k, x_k in enumerate(windows):
               token = forward_step(x_k, seq_idx=k)
        3. reset_cache()

    Args:
        d_input: pooled window vector dim.
        d_model: Mamba2 internal dim. Default 256.
        n_layers: number of stacked Mamba2 blocks. Default 1.
        llm_hidden: output dim matching LLM embedding size.
                    Default 3584 (Qwen2-VL-7B).
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
            raise ImportError(
                "mamba_ssm is not installed. "
                "Install it via: pip install mamba-ssm"
            )

        self.d_model = d_model
        self.n_layers = n_layers

        # Compressed window vector -> Mamba2 dim
        self.in_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
        )

        # Explicit Mamba2 blocks with residual.
        # Same pattern as StreamMind VideoMamba but using Mamba2 directly
        # instead of going through create_block.
        self.blocks = nn.ModuleList([
            Mamba2(
                d_model=d_model,
                d_state=64,
                d_conv=4,
                expand=2,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Final projection into LLM embedding space
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, llm_hidden)

        # Inference cache (one per block, allocated per-video for streaming)
        self._cache = None

    # ------------------------------------------------------------------
    # Training: all windows at once
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch forward over all windows of a video.

        Args:
            x: [B, T, d_input] — T pooled window vectors.

        Returns:
            ssm_tokens: [B, T, llm_hidden]
        """
        h = self.in_proj(x)           # [B, T, d_model]

        for blk, norm in zip(self.blocks, self.norms):
            residual = h
            h = norm(h)
            h = blk(h)                # [B, T, d_model]
            h = h + residual          # residual connection

        h = self.out_norm(h)
        return self.out_proj(h)       # [B, T, llm_hidden]

    # ------------------------------------------------------------------
    # Streaming: one window at a time with stateful Mamba2
    # ------------------------------------------------------------------
    def allocate_cache(self, batch_size: int, max_windows: int) -> None:
        """Pre-allocate conv_state and ssm_state for all Mamba2 blocks.

        Must be called once per video before any forward_step() calls.

        Args:
            batch_size: typically 1 for online inference.
            max_windows: upper bound on flush count for this video.
        """
        self._cache = {}
        for i, blk in enumerate(self.blocks):
            conv_state, ssm_state = blk.allocate_inference_cache(
                batch_size, max_windows
            )
            self._cache[i] = (conv_state, ssm_state)

    def forward_step(
        self, x: torch.Tensor, seq_idx: int
    ) -> torch.Tensor:
        """Process a single window in streaming mode using Mamba2.step().

        Requires allocate_cache() to have been called first.

        Uses Mamba2's native step() method which correctly updates
        conv_state and ssm_state internally via the select_state_update
        Triton kernel.

        Args:
            x: [B, 1, d_input] — one pooled window vector.
            seq_idx: position index of this window (0-based).

        Returns:
            ssm_token: [B, 1, llm_hidden]
        """
        if self._cache is None:
            raise RuntimeError(
                "Inference cache not allocated. Call allocate_cache() first."
            )

        h = self.in_proj(x)           # [B, 1, d_model]

        for i, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
            residual = h
            h = norm(h)
            conv_state, ssm_state = self._cache[i]
            # Mamba2.step(): single-token streaming with internal state update
            h, conv_state, ssm_state = blk.step(
                h, conv_state, ssm_state
            )
            self._cache[i] = (conv_state, ssm_state)
            h = h + residual

        h = self.out_norm(h)
        return self.out_proj(h)       # [B, 1, llm_hidden]

    def reset_cache(self) -> None:
        """Release inference cache (call at end of video)."""
        self._cache = None

    def get_cache(self, detach: bool = False):
        """Return a copy of the current per-block (conv_state, ssm_state)."""
        if self._cache is None:
            return None
        out = {}
        for i, (cs, ss) in self._cache.items():
            if detach:
                out[i] = (cs.detach().clone(), ss.detach().clone())
            else:
                out[i] = (cs.clone(), ss.clone())
        return out

    def set_cache(self, cache: dict | None) -> None:
        """Restore per-block (conv_state, ssm_state) from a previous get_cache()."""
        if cache is None:
            self._cache = None
            return
        # move to the device of the first block's parameter
        dev = next(self.blocks[0].parameters()).device
        self._cache = {}
        for i, (cs, ss) in cache.items():
            self._cache[i] = (cs.to(dev), ss.to(dev))
