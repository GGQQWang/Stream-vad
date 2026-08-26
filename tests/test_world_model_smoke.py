"""GPU smoke tests for the world-model branch (SSM state-change token).

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


def _rand_target():
    return torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))


def test_decoder_produces_per_position_ce():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    h_t = torch.randn(256)
    tgt = _rand_target()

    ce = branch.forward_once(C_t, mask, h_t, None, codebook, tgt, logit_chunk_size=32)
    assert ce.ndim == 0
    ce.backward()
    assert branch.visual_proj.weight.grad is not None
    assert branch.change_proj.weight.grad is not None
    assert branch.change_proj.bias is None, "change_proj must have no bias"
    print("test 1 OK: per-position 392-way CE, gradients flow, change_proj bias-free")


def test_causal_mask_is_upper_triangular():
    T = IBQ_TOKENS_PER_FRAME
    m = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    assert torch.isinf(m[0, 1]) and not torch.isinf(m[1, 0])
    print("test 2 OK: causal mask upper-triangular")


def test_chunked_ce_equals_full_ce():
    tgt = _rand_target()
    logits = torch.randn(IBQ_TOKENS_PER_FRAME, IBQ_CODEBOOK_SIZE)
    full = F.cross_entropy(logits, tgt, reduction="mean")
    chunk_sum = sum(
        F.cross_entropy(logits[s:s + 32], tgt[s:s + 32], reduction="sum")
        for s in range(0, IBQ_TOKENS_PER_FRAME, 32)
    )
    assert abs(chunk_sum / IBQ_TOKENS_PER_FRAME - full) < 1e-5
    print("test 3 OK: chunked CE equals full CE")


def test_condition_is_the_state_change():
    """Same delta (h_t - h_prev) with different absolute states must
    give identical CE; the condition is the change, not the state."""
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    tgt = _rand_target()

    d = torch.randn(256)
    h1, h2 = torch.randn(256), torch.randn(256)
    h3 = h1 + torch.randn(256)
    with torch.no_grad():
        ce_a = branch.forward_once(C_t, mask, h1 + d, h1, codebook, tgt)
        ce_b = branch.forward_once(C_t, mask, h3 + d, h3, codebook, tgt)
    assert abs(ce_a.item() - ce_b.item()) < 1e-4
    # h_prev=None behaves like delta = zeros: identical to h_prev == h_t
    with torch.no_grad():
        ce_c = branch.forward_once(C_t, mask, h1, None, codebook, tgt)
        ce_d = branch.forward_once(C_t, mask, h1, h1, codebook, tgt)
    assert abs(ce_c.item() - ce_d.item()) < 1e-4
    print("test 4 OK: condition is h_t - h_prev (state change)")


def test_zero_change_token_is_truly_zero():
    """zero_change must bypass change_proj entirely: CE identical for
    any h_t, and different from the normal conditioned forward."""
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    tgt = _rand_target()

    with torch.no_grad():
        ce_z1 = branch.forward_once(
            C_t, mask, torch.randn(256), None, codebook, tgt, zero_change=True)
        ce_z2 = branch.forward_once(
            C_t, mask, torch.randn(256), torch.randn(256), codebook, tgt,
            zero_change=True)
        ce_n = branch.forward_once(
            C_t, mask, torch.randn(256), None, codebook, tgt)
    assert abs(ce_z1.item() - ce_z2.item()) < 1e-4, "zero token must be independent of h"
    assert abs(ce_z1.item() - ce_n.item()) > 1e-6
    print("test 5 OK: zero-change token is genuinely all-zero")


def test_joint_gradient_reaches_h_through_change_token():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    h_t = torch.randn(256, requires_grad=True)
    h_prev = torch.randn(256, requires_grad=True)
    tgt = _rand_target()

    ce = branch.forward_once(C_t, mask, h_t, h_prev, codebook, tgt)
    ce.backward()
    assert h_t.grad is not None and h_t.grad.abs().sum() > 0
    assert h_prev.grad is not None and h_prev.grad.abs().sum() > 0
    print("test 6 OK: IBQ CE -> change token -> h_t / h_prev gradient path")


def test_zero_delta_gives_exactly_zero_change_token():
    """With h_prev=None (delta_h = 0) and no bias, the change token
    must be exactly zero."""
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    assert branch.change_proj.bias is None
    h_t = torch.randn(256)
    delta = torch.zeros_like(h_t)
    token = branch.change_proj(delta)
    assert token.abs().max().item() == 0.0, "change_proj(0) must be exactly 0"
    print("test 5b OK: delta_h=0 -> change token exactly zero")


def test_cross_chunk_prev_h_and_video_end_clear():
    from pipeline_stage1 import _clear_finished_states, _world_model_loss

    class _FakeModel:
        def __init__(self):
            self.world_branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
            self.ibq_codebook = torch.randn(
                IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
            self.debug_state = False

    class _FakeIBQ:
        def __init__(self, n_windows):
            self.n_windows = n_windows

        def get(self, vid, window_idx, frame_idx):
            if window_idx >= self.n_windows:
                raise IndexError
            return torch.randint(0, IBQ_CODEBOOK_SIZE, (IBQ_TOKENS_PER_FRAME,))

    model = _FakeModel()
    ibq = _FakeIBQ(n_windows=16)
    B, W, R = 1, 4, 8
    sf = torch.randn(B, W, R, 3584)
    sm = torch.ones(B, W, R, dtype=torch.bool)
    valid = torch.ones(B, W, dtype=torch.bool)
    prev_cache: dict = {}

    h1 = torch.randn(B, W, 256, requires_grad=True)
    batch1 = {"chunk_start": [0], "video_id": ["v1"],
              "spatial_features": sf, "spatial_mask": sm}
    _world_model_loss(model, ibq, batch1, valid, valid, h1, sf, sm,
                      prev_cache, 1, 16, 32, False, detach_states=False)
    assert "v1" in prev_cache, "prev_h not cached after chunk 1"
    assert torch.allclose(prev_cache["v1"], h1[0, -1])
    assert prev_cache["v1"].requires_grad is False, (
        "cross-chunk prev_h must be detached (TBPTT)"
    )

    h2 = torch.randn(B, W, 256, requires_grad=True)
    batch2 = {"chunk_start": [4], "video_id": ["v1"],
              "spatial_features": sf, "spatial_mask": sm}
    _world_model_loss(model, ibq, batch2, valid, valid, h2, sf, sm,
                      prev_cache, 1, 16, 32, False, detach_states=False)
    assert torch.allclose(prev_cache["v1"], h2[0, -1]), "prev_h not updated to chunk 2"
    assert prev_cache["v1"].requires_grad is False

    ssm_cache = {"v1": object()}
    _clear_finished_states(
        model, {"video_id": ["v1"], "is_last_chunk": [True]},
        ssm_cache, prev_cache,
    )
    assert "v1" not in prev_cache, "prev_h not cleared at video end"
    assert "v1" not in ssm_cache
    print("test 7 OK: cross-chunk prev_h reuse + video-end clearing")


def test_grid_shape_assert():
    assert IBQ_GRID_ROWS * IBQ_GRID_COLS == IBQ_TOKENS_PER_FRAME
    print("test 8 OK: grid shape consistent")


if __name__ == "__main__":
    test_decoder_produces_per_position_ce()
    test_causal_mask_is_upper_triangular()
    test_chunked_ce_equals_full_ce()
    test_condition_is_the_state_change()
    test_zero_change_token_is_truly_zero()
    test_zero_delta_gives_exactly_zero_change_token()
    test_joint_gradient_reaches_h_through_change_token()
    test_cross_chunk_prev_h_and_video_end_clear()
    test_grid_shape_assert()
    print("ALL WORLD-MODEL SMOKE TESTS PASSED")
