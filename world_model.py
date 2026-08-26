"""World-model auxiliary branch (training-only).

Architecture:

    current window spatial tokens  C_t [R, 3584]      (SpatialTokenCompressor
    output, pre-mean-pooling, from feature cache v3)
    SSM internal temporal hidden  h_t [d_ssm]          (out_norm output,
    pre-out_proj; exposed via SSMBlock.forward_chunk(return_internal=True))

        C_t --visual_proj--> [R, D]
        delta_h = h_t - h_{t-1}  (state CHANGE, historical — not a
        future prediction)
        change_token = change_proj(delta_h)  [D]

        memory = concat([visual_tokens; change_token]) -> [R+1, D]

    An autoregressive TransformerDecoder (teacher forcing, causal mask,
    2D positional embeddings) predicts the future frame's IBQ token
    sequence; each of the 392 positions gets its OWN [131072] logits
    via a dot product with the frozen IBQ codebook.

    A zero-change baseline forward (change token replaced by an
    all-zero decoder token, no_grad) quantifies how much the state
    change contributes: ibq_change_gain = CE_zero - CE_change.

Only the module definition and the per-window forward live here; the
training loop in pipeline_stage1.py owns data plumbing (including the
cross-chunk h_{t-1} cache) and loss accumulation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ibq_utils import (
    IBQ_CODEBOOK_SIZE,
    IBQ_CODE_EMBED_DIM,
    IBQ_TOKENS_PER_FRAME,
)


# IBQ grid is 28 rows x 14 cols at 448x224 (16x downsampling)
IBQ_GRID_ROWS = 28
IBQ_GRID_COLS = 14

assert IBQ_GRID_ROWS * IBQ_GRID_COLS == IBQ_TOKENS_PER_FRAME, (
    f"IBQ grid {IBQ_GRID_ROWS}x{IBQ_GRID_COLS} != {IBQ_TOKENS_PER_FRAME} tokens"
)


class WorldModelBranch(nn.Module):
    """Autoregressive IBQ visual decoder conditioned on SSM state change.

    Training-only; discarded at inference.  All tensors flow per
    window, never as a shared batch over windows.
    """

    def __init__(
        self,
        llm_hidden: int = 3584,
        d_ssm: int = 256,
        decoder_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        assert IBQ_CODE_EMBED_DIM == decoder_dim, (
            f"IBQ_CODE_EMBED_DIM={IBQ_CODE_EMBED_DIM} must equal "
            f"decoder_dim={decoder_dim}"
        )
        self.d_ssm = d_ssm
        self.decoder_dim = decoder_dim

        # --- current visual context projection ---
        self.visual_proj = nn.Linear(llm_hidden, decoder_dim)

        # --- state-change projection: delta_h = h_t - h_{t-1} ---
        # no bias: the first window of a video has delta_h = 0 and its
        # change token must be exactly zero
        self.change_proj = nn.Linear(d_ssm, decoder_dim, bias=False)

        # --- decoder input tokens ---
        self.world_bos = nn.Parameter(torch.randn(1, decoder_dim) * 0.02)
        self.row_pos = nn.Parameter(torch.randn(IBQ_GRID_ROWS, decoder_dim) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(IBQ_GRID_COLS, decoder_dim) * 0.02)

        # --- autoregressive decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- output projection into codebook embedding space ---
        self.output_proj = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
        )

    # ------------------------------------------------------------------
    def _position_ids(self, device: torch.device) -> torch.Tensor:
        """Row-major 2D positions for the 392-token IBQ grid."""
        rows = torch.arange(IBQ_GRID_ROWS, device=device).repeat_interleave(IBQ_GRID_COLS)
        cols = torch.arange(IBQ_GRID_COLS, device=device).repeat(IBQ_GRID_ROWS)
        return self.row_pos[rows] + self.col_pos[cols]          # [392, D]

    # ------------------------------------------------------------------
    def decode_ce(
        self,
        spatial_tokens: torch.Tensor,       # [R, 3584]
        spatial_mask: torch.Tensor,         # [R] bool
        change_token: torch.Tensor,         # [D] decoder-space token
        ibq_codebook: torch.Tensor,         # [V, E] frozen
        tgt: torch.Tensor,                  # [392] long
        logit_chunk_size: int = 32,
    ) -> torch.Tensor:
        """Autoregressive teacher-forced CE for ONE future frame.

        Returns the mean per-token CE over all 392 positions.  Logits
        are computed in chunks of ``logit_chunk_size`` positions to
        bound peak memory.
        """
        T = IBQ_TOKENS_PER_FRAME
        assert tgt.numel() == T, f"target has {tgt.numel()} tokens, expected {T}"
        assert ibq_codebook.shape == (IBQ_CODEBOOK_SIZE, self.decoder_dim), (
            f"codebook shape {tuple(ibq_codebook.shape)}"
        )
        assert change_token.shape[-1] == self.decoder_dim, (
            f"change token dim {change_token.shape[-1]}, "
            f"expected {self.decoder_dim}"
        )
        device = spatial_tokens.device

        # --- memory: current visual tokens + state-change token ---
        vis = self.visual_proj(spatial_tokens)                 # [R, D]
        vis = vis[spatial_mask.bool()]                          # [R_real, D]
        change = change_token.reshape(1, self.decoder_dim)      # [1, D]
        memory = torch.cat([vis, change], dim=0)                # [R_real+1, D]

        # --- teacher-forced decoder input: [BOS, emb(y0) ... emb(y_390)] ---
        emb = ibq_codebook[tgt[:-1]]                            # [T-1, D]
        decoder_input = torch.cat([self.world_bos, emb], dim=0)  # [T, D]
        decoder_input = decoder_input + self._position_ids(device)

        tgt_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1,
        )

        out = self.decoder(
            tgt=decoder_input.unsqueeze(0),
            memory=memory.unsqueeze(0),
            tgt_mask=tgt_mask,
        )[0]                                                    # [T, D]
        z = self.output_proj(out)                               # [T, D]

        # --- chunked per-position CE against the frozen codebook ---
        # mathematically identical to one [T, V] CE: the loss is a sum
        # over positions, so summing per-chunk sums is exact
        total_ce: Optional[torch.Tensor] = None
        for start in range(0, T, logit_chunk_size):
            end = min(start + logit_chunk_size, T)
            logits = F.linear(z[start:end], ibq_codebook)       # [C, V]
            ce = F.cross_entropy(logits, tgt[start:end], reduction="sum")
            total_ce = ce if total_ce is None else total_ce + ce
        return total_ce / T

    # ------------------------------------------------------------------
    def forward_once(
        self,
        spatial_tokens: torch.Tensor,
        spatial_mask: torch.Tensor,
        h_t: torch.Tensor,                  # [d_ssm]
        h_prev: Optional[torch.Tensor],     # [d_ssm] or None (first window)
        ibq_codebook: torch.Tensor,
        tgt: torch.Tensor,
        logit_chunk_size: int = 32,
        zero_change: bool = False,
    ) -> torch.Tensor:
        """One window's world forward.  Returns the mean IBQ CE.

        The decoder condition is the historical state CHANGE
        ``h_t - h_{t-1}`` projected through ``change_proj`` — not a
        future prediction.  With ``zero_change=True`` the decoder
        token is replaced by an all-zero token (bypassing
        ``change_proj`` entirely, so its bias does not leak in).
        """
        if h_prev is None:
            delta_h = torch.zeros_like(h_t)
        else:
            delta_h = h_t - h_prev
        if zero_change:
            # genuinely all-zero decoder token; change_proj is bypassed
            # entirely so its bias cannot leak into the baseline
            change_token = torch.zeros(
                1, self.decoder_dim, device=h_t.device, dtype=h_t.dtype,
            )
        else:
            change_token = self.change_proj(delta_h)            # [D]
        return self.decode_ce(
            spatial_tokens, spatial_mask, change_token,
            ibq_codebook, tgt, logit_chunk_size,
        )
