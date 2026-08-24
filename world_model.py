"""World-model auxiliary branch (training-only).

Architecture:

    current window spatial tokens  C_t [R, 3584]      (SpatialTokenCompressor
    output, pre-mean-pooling, from feature cache v3)
    SSM internal temporal hidden  h_t [d_ssm]          (out_norm output,
    pre-out_proj; exposed via SSMBlock.forward_chunk(return_internal=True))

        C_t --visual_proj--> [R, D]
        h_t --delta_predictor--> delta_pred [d_ssm]
                                 (supervised by (h_{t+k} - h_t).detach(),
                                  SmoothL1)
        delta_pred --delta_proj--> [1, D]

        memory = concat([visual_tokens; delta_token])  -> [R+1, D]

    An autoregressive TransformerDecoder (teacher forcing, causal mask,
    2D positional embeddings) predicts the future frame's IBQ token
    sequence; each of the 392 positions gets its OWN [131072] logits
    via a dot product with the frozen IBQ codebook.

    A zero-delta baseline forward (delta_token replaced by zeros,
    no_grad) quantifies how much the predicted dynamics contribute.

Only the module definition and the per-window forward live here; the
training loop in pipeline_stage1.py owns data plumbing and loss
accumulation.
"""

from typing import Optional, Tuple

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
    """Delta predictor + autoregressive IBQ visual decoder.

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

        # --- temporal delta predictor: h_t -> predicted h_{t+k} - h_t ---
        self.delta_predictor = nn.Sequential(
            nn.Linear(d_ssm, d_ssm),
            nn.GELU(),
            nn.LayerNorm(d_ssm),
            nn.Linear(d_ssm, d_ssm),
        )

        # --- projections into the decoder space ---
        self.visual_proj = nn.Linear(llm_hidden, decoder_dim)
        self.delta_proj = nn.Sequential(
            nn.Linear(d_ssm, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )

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

    def delta_forward(
        self,
        h_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predicted temporal delta from the SSM internal hidden."""
        assert h_t.shape[-1] == self.d_ssm, (
            f"h_t dim mismatch: got {h_t.shape[-1]}, expected {self.d_ssm}"
        )
        return self.delta_predictor(h_t)                        # [..., d_ssm]

    # ------------------------------------------------------------------
    def decode_ce(
        self,
        spatial_tokens: torch.Tensor,       # [R, 3584]
        spatial_mask: torch.Tensor,         # [R] bool
        delta_pred: torch.Tensor,           # [d_ssm]
        ibq_codebook: torch.Tensor,         # [V, E] frozen
        tgt: torch.Tensor,                  # [392] long
        logit_chunk_size: int = 32,
    ) -> torch.Tensor:
        """Autoregressive teacher-forced CE for ONE future frame.

        Returns the mean per-token CE over all 392 positions.  Logits
        are computed in chunks of ``logit_chunk_size`` positions to
        bound peak memory (a full [392, 131072] tensor is ~205MB fp32).
        """
        T = IBQ_TOKENS_PER_FRAME
        assert tgt.numel() == T, f"target has {tgt.numel()} tokens, expected {T}"
        assert ibq_codebook.shape == (IBQ_CODEBOOK_SIZE, self.decoder_dim), (
            f"codebook shape {tuple(ibq_codebook.shape)}"
        )
        device = spatial_tokens.device

        # --- memory: current visual tokens + delta token ---
        vis = self.visual_proj(spatial_tokens)                 # [R, D]
        vis = vis[spatial_mask.bool()]                          # [R_real, D]
        delta_token = self.delta_proj(delta_pred).unsqueeze(0)  # [1, D]
        memory = torch.cat([vis, delta_token], dim=0)           # [R_real+1, D]

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
        h_t: torch.Tensor,
        ibq_codebook: torch.Tensor,
        tgt: torch.Tensor,
        delta_target: Optional[torch.Tensor] = None,
        logit_chunk_size: int = 32,
        zero_delta: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One window's world forward.

        Returns ``(ibq_ce, loss_delta)``.  With ``zero_delta=True`` the
        delta token is replaced by zeros (shortcut baseline, always
        run under no_grad by the caller).
        """
        delta_pred = self.delta_forward(h_t)
        if zero_delta:
            delta_pred = torch.zeros_like(delta_pred)

        ibq_ce = self.decode_ce(
            spatial_tokens, spatial_mask, delta_pred,
            ibq_codebook, tgt, logit_chunk_size,
        )
        if delta_target is None:
            loss_delta = ibq_ce.new_zeros(())
        else:
            loss_delta = F.smooth_l1_loss(delta_pred, delta_target)
        return ibq_ce, loss_delta
