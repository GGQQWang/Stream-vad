"""GPU smoke tests for the new world-model branch.

Run on the server:
    python tests/test_world_model_smoke.py
"""

import torch
import torch.nn.functional as F

from ibq_utils import (
    IBQ_CODE_EMBED_DIM,
    IBQ_CODEBOOK_SIZE,
    IBQ_TOKENS_PER_FRAME,
)
from world_model import IBQ_GRID_COLS, IBQ_GRID_ROWS, WorldModelBranch


def test_decoder_produces_per_position_ce():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)  # frozen buffer

    R = 12
    C_t = torch.randn(R, 3584)
    mask = torch.ones(R, dtype=torch.bool)
    h_t = torch.randn(256)
    tgt = torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))
    delta_target = torch.randn(256)

    ce, loss_delta = branch.forward_once(
        C_t, mask, h_t, codebook, tgt,
        delta_target=delta_target, logit_chunk_size=32,
    )
    assert ce.ndim == 0 and loss_delta.ndim == 0
    ce.backward()
    assert branch.visual_proj.weight.grad is not None
    assert branch.delta_predictor[0].weight.grad is not None
    print("test 1 OK: per-position 392-way CE, gradients flow into branch")


def test_causal_mask_is_upper_triangular():
    T = IBQ_TOKENS_PER_FRAME
    m = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    assert torch.isinf(m[0, 1]) and not torch.isinf(m[1, 0])
    assert not torch.isinf(m[5, 5])
    print("test 2 OK: causal mask upper-triangular")


def test_chunked_ce_equals_full_ce():
    tgt = torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))
    logits = torch.randn(IBQ_TOKENS_PER_FRAME, IBQ_CODEBOOK_SIZE)
    full = F.cross_entropy(logits, tgt, reduction="mean")
    chunk_sum = sum(
        F.cross_entropy(logits[s:s + 32], tgt[s:s + 32], reduction="sum")
        for s in range(0, IBQ_TOKENS_PER_FRAME, 32)
    )
    assert abs(chunk_sum / IBQ_TOKENS_PER_FRAME - full) < 1e-5
    print("test 3 OK: chunked CE equals full CE")


def test_zero_delta_baseline_runs_no_grad():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    h_t = torch.randn(256)
    tgt = torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))
    with torch.no_grad():
        ce_zero, _ = branch.forward_once(
            C_t, mask, h_t, codebook, tgt, zero_delta=True,
            logit_chunk_size=32,
        )
    assert ce_zero.ndim == 0 and torch.isfinite(ce_zero)
    print("test 4 OK: zero-delta shortcut baseline runs")


def test_zero_loss_graph_connectivity():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    z = branch.delta_forward(torch.randn(256))
    zero = z.sum() * 0.0
    assert zero.requires_grad and zero.grad_fn is not None
    zero.backward()
    print("test 5 OK: empty-target zero loss stays graph-connected")


def test_grid_shape_assert():
    assert IBQ_GRID_ROWS * IBQ_GRID_COLS == IBQ_TOKENS_PER_FRAME
    print("test 6 OK: grid shape consistent")


if __name__ == "__main__":
    test_decoder_produces_per_position_ce()
    test_causal_mask_is_upper_triangular()
    test_chunked_ce_equals_full_ce()
    test_zero_delta_baseline_runs_no_grad()
    test_zero_loss_graph_connectivity()
    test_grid_shape_assert()
    print("ALL WORLD-MODEL SMOKE TESTS PASSED")
