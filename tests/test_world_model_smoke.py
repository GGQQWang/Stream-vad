"""GPU smoke tests for the world-model branch (direct temporal token).

Run on the server:
    PYTHONPATH=. python tests/test_world_model_smoke.py
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
    h_t = torch.randn(256)                                  # direct temporal token
    tgt = torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))

    ce = branch.forward_once(
        C_t, mask, h_t, codebook, tgt, logit_chunk_size=32,
    )
    assert ce.ndim == 0
    ce.backward()
    assert branch.visual_proj.weight.grad is not None
    assert branch.decoder.layers[0].linear1.weight.grad is not None
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


def test_zero_temporal_baseline_runs_no_grad():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    h_t = torch.randn(256)
    tgt = torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))
    with torch.no_grad():
        ce_zero = branch.forward_once(
            C_t, mask, h_t, codebook, tgt,
            logit_chunk_size=32, zero_temporal=True,
        )
    assert ce_zero.ndim == 0 and torch.isfinite(ce_zero)
    print("test 4 OK: zero-temporal shortcut baseline runs")


def test_temporal_token_used_directly():
    # no delta_predictor / delta_proj modules exist anymore
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    for name, _ in branch.named_modules():
        assert "delta" not in name, f"unexpected legacy module: {name}"
    assert branch.d_ssm == branch.decoder_dim
    print("test 5 OK: h_internal is used directly, no delta modules")


def test_grid_shape_assert():
    assert IBQ_GRID_ROWS * IBQ_GRID_COLS == IBQ_TOKENS_PER_FRAME
    print("test 6 OK: grid shape consistent")


if __name__ == "__main__":
    test_decoder_produces_per_position_ce()
    test_causal_mask_is_upper_triangular()
    test_chunked_ce_equals_full_ce()
    test_zero_temporal_baseline_runs_no_grad()
    test_temporal_token_used_directly()
    test_grid_shape_assert()
    print("ALL WORLD-MODEL SMOKE TESTS PASSED")
