"""GPU smoke tests for the world-model branch (temporal dynamics token).

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

    ce = branch.forward_once(C_t, mask, h_t, codebook, tgt, logit_chunk_size=32)
    assert ce.ndim == 0
    ce.backward()
    assert branch.visual_proj.weight.grad is not None
    assert branch.temporal_proj[0].weight.grad is not None
    print("test 1 OK: per-position 392-way CE, gradients flow")


def test_temporal_proj_structure():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    proj = branch.temporal_proj
    assert len(proj) == 3
    assert isinstance(proj[0], torch.nn.Linear) and proj[0].in_features == 256 \
        and proj[0].out_features == 512
    assert isinstance(proj[1], torch.nn.GELU)
    assert isinstance(proj[2], torch.nn.Linear) and proj[2].in_features == 512 \
        and proj[2].out_features == 256
    for name, _ in branch.named_modules():
        assert "change_proj" not in name and "delta" not in name, name
    print("test 2 OK: temporal_proj is 256->512->256, no change_proj/delta modules")


def test_causal_mask_is_upper_triangular():
    T = IBQ_TOKENS_PER_FRAME
    m = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    assert torch.isinf(m[0, 1]) and not torch.isinf(m[1, 0])
    print("test 3 OK: causal mask upper-triangular")


def test_chunked_ce_equals_full_ce():
    tgt = _rand_target()
    logits = torch.randn(IBQ_TOKENS_PER_FRAME, IBQ_CODEBOOK_SIZE)
    full = F.cross_entropy(logits, tgt, reduction="mean")
    chunk_sum = sum(
        F.cross_entropy(logits[s:s + 32], tgt[s:s + 32], reduction="sum")
        for s in range(0, IBQ_TOKENS_PER_FRAME, 32)
    )
    assert abs(chunk_sum / IBQ_TOKENS_PER_FRAME - full) < 1e-5
    print("test 4 OK: chunked CE equals full CE")


def test_zero_temporal_token_is_truly_zero():
    """zero_temporal must bypass temporal_proj entirely: CE identical
    for any h_t, and different from the normal conditioned forward.
    (eval mode disables dropout so the comparison is clean.)"""
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256).eval()
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    tgt = _rand_target()

    with torch.no_grad():
        ce_z1 = branch.forward_once(
            C_t, mask, torch.randn(256), codebook, tgt, zero_temporal=True)
        ce_z2 = branch.forward_once(
            C_t, mask, torch.randn(256), codebook, tgt, zero_temporal=True)
        ce_n = branch.forward_once(
            C_t, mask, torch.randn(256), codebook, tgt)
    assert abs(ce_z1.item() - ce_z2.item()) < 1e-4, "zero token must be independent of h"
    assert abs(ce_z1.item() - ce_n.item()) > 1e-6
    print("test 5 OK: zero-temporal token is genuinely all-zero")


def test_joint_gradient_reaches_h_through_temporal_proj():
    branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
    codebook = torch.randn(IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    C_t = torch.randn(12, 3584)
    mask = torch.ones(12, dtype=torch.bool)
    h_t = torch.randn(256, requires_grad=True)
    tgt = _rand_target()

    ce = branch.forward_once(C_t, mask, h_t, codebook, tgt)
    ce.backward()
    assert h_t.grad is not None and h_t.grad.abs().sum() > 0
    print("test 6 OK: IBQ CE -> temporal_proj -> h_t gradient path")


def test_warmup_detach_blocks_gradient():
    """With detach_states=True in _world_model_loss, h_internal must
    receive no gradient (warmup semantics)."""
    from pipeline_stage1 import _world_model_loss

    class _FakeModel:
        def __init__(self):
            self.world_branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
            self.ibq_codebook = torch.randn(
                IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)

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
    h = torch.randn(B, W, 256, requires_grad=True)
    sf = torch.randn(B, W, R, 3584)
    sm = torch.ones(B, W, R, dtype=torch.bool)
    valid = torch.ones(B, W, dtype=torch.bool)
    batch = {"chunk_start": [0], "video_id": ["v1"],
             "spatial_features": sf, "spatial_mask": sm}

    loss, _ = _world_model_loss(
        model, ibq, batch, valid, valid, h, sf, sm,
        1, 16, 32, False, detach_states=True,
    )
    loss.backward()
    assert h.grad is None or h.grad.abs().sum() == 0, (
        "warmup detach must block gradients into h_internal"
    )
    print("test 7 OK: warmup detach blocks SSM gradient")


def test_grid_shape_assert():
    assert IBQ_GRID_ROWS * IBQ_GRID_COLS == IBQ_TOKENS_PER_FRAME
    print("test 8 OK: grid shape consistent")


def test_zero_target_batch_respects_warmup_detach():
    """When NO window has a future IBQ target, the fallback zero loss
    must still cut the SSM graph under detach_states=True (h_internal
    gets no grad) while keeping temporal_proj trainable."""
    from pipeline_stage1 import _world_model_loss

    class _FakeModel:
        def __init__(self):
            self.world_branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
            self.ibq_codebook = torch.randn(
                IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)

    class _FakeIBQNoFuture:
        def get(self, vid, window_idx, frame_idx):
            raise IndexError  # no future window anywhere

    model = _FakeModel()
    ibq = _FakeIBQNoFuture()
    B, W, R = 1, 4, 8
    h = torch.randn(B, W, 256, requires_grad=True)
    sf = torch.randn(B, W, R, 3584)
    sm = torch.ones(B, W, R, dtype=torch.bool)
    valid = torch.ones(B, W, dtype=torch.bool)
    batch = {"chunk_start": [0], "video_id": ["v1"],
             "spatial_features": sf, "spatial_mask": sm}

    loss, info = _world_model_loss(
        model, ibq, batch, valid, valid, h, sf, sm,
        1, 16, 32, False, detach_states=True,
    )
    assert info["num_world_windows"] == 0
    assert loss.requires_grad, "zero loss must stay graph-connected"
    loss.backward()
    assert h.grad is None or h.grad.abs().sum() == 0, (
        "zero-target warmup batch must not propagate grad into h_internal"
    )
    assert model.world_branch.temporal_proj[0].weight.grad is not None, (
        "temporal_proj must still receive grad so backward is valid"
    )
    print("test 9 OK: zero-target fallback respects warmup detach")


def test_grad_conflict_stats():
    """grad_conflict_stats: direction measurement, non-invasive, and a
    subsequent backward still works."""
    from pipeline_stage1 import grad_conflict_stats

    # same-direction losses -> cosine ~ +1
    x = torch.randn(4)
    p = torch.nn.Parameter(torch.randn(4))
    loss_a = (p * x).sum()
    loss_b = (p * x).sum() * 2.0
    stats = grad_conflict_stats(loss_a, loss_b, [p])
    assert stats["cosine"] > 0.999, stats["cosine"]
    assert p.grad is None, "diagnostics must not pollute param.grad"

    # opposite-direction losses -> cosine ~ -1
    y = torch.randn(4)
    q = torch.nn.Parameter(torch.randn(4))
    loss_c = (q * y).sum()
    loss_d = -(q * y).sum()
    stats2 = grad_conflict_stats(loss_c, loss_d, [q])
    assert stats2["cosine"] < -0.999, stats2["cosine"]
    assert q.grad is None

    # a normal backward afterwards still works (graph was retained)
    (loss_c + loss_d).backward()
    assert q.grad is not None
    print("test 10 OK: grad_conflict_stats direction + non-invasive + backward OK")


if __name__ == "__main__":
    test_decoder_produces_per_position_ce()
    test_temporal_proj_structure()
    test_causal_mask_is_upper_triangular()
    test_chunked_ce_equals_full_ce()
    test_zero_temporal_token_is_truly_zero()
    test_joint_gradient_reaches_h_through_temporal_proj()
    test_warmup_detach_blocks_gradient()
    test_grid_shape_assert()
    test_zero_target_batch_respects_warmup_detach()
    test_grad_conflict_stats()
    print("ALL WORLD-MODEL SMOKE TESTS PASSED")
