"""GPU smoke tests for the world-model branch (temporal dynamics token).

Run on the server:
    PYTHONPATH=. python tests/test_world_model_smoke.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import types

from ibq_utils import (
    IBQ_CODE_EMBED_DIM,
    IBQ_CODEBOOK_SIZE,
    IBQ_TOKENS_PER_FRAME,
)
from world_model import IBQ_GRID_COLS, IBQ_GRID_ROWS, WorldModelBranch


def _install_pipeline_import_stubs():
    tb_mod = types.ModuleType("torch.utils.tensorboard")
    tb_mod.SummaryWriter = object
    sys.modules.setdefault("torch.utils.tensorboard", tb_mod)

    tr_mod = types.ModuleType("transformers")
    tr_mod.AutoTokenizer = object
    tr_mod.Qwen2VLForConditionalGeneration = nn.Module
    tr_mod.Qwen2VLProcessor = object
    tr_mod.get_linear_schedule_with_warmup = lambda *args, **kwargs: None
    tr_mod.set_seed = lambda *args, **kwargs: None
    sys.modules.setdefault("transformers", tr_mod)

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda model, *args, **kwargs: model
    sys.modules.setdefault("peft", peft_mod)


def _import_pipeline_stage1():
    _install_pipeline_import_stubs()
    import pipeline_stage1 as pipe
    return pipe


class _FakeSSM(nn.Module):
    def __init__(self, d_input, d_model=256, n_layers=1, llm_hidden=None):
        super().__init__()
        out_dim = d_input if llm_hidden is None else llm_hidden
        self.in_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
        )
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward_chunk(self, x, state=None, return_internal=False):
        h = self.in_proj(x)
        out = self.out_proj(h)
        new_state = {0: h[:, -1:].detach()}
        if return_internal:
            return out, new_state, h
        return out, new_state


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Identity()


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
    _world_model_loss = _import_pipeline_stage1()._world_model_loss

    class _FakeModel:
        def __init__(self):
            self.world_branch = WorldModelBranch(llm_hidden=3584, d_ssm=256)
            self.ibq_codebook = torch.randn(
                IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)

    class _FakeIBQ:
        def __init__(self, n_windows):
            self.n_windows = n_windows

        def num_windows(self, vid):
            return self.n_windows

        def valid_frame_count(self, vid, window_idx):
            if window_idx >= self.n_windows:
                raise IndexError
            return 16

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


class _TinyWorldBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_proj = nn.Linear(2, 2)

    def forward_once(
        self, C_t, m_t, h_t, codebook, tgt, logit_chunk_size=32,
        zero_temporal=False,
    ):
        return h_t.sum() * 0.0 + tgt.float().mean() * 0.0 + 1.0


class _TinyWorldModel:
    def __init__(self):
        self.world_branch = _TinyWorldBranch()
        self.ibq_codebook = torch.empty(0)


def _tiny_world_inputs():
    B, W, R, H = 1, 1, 1, 2
    h = torch.randn(B, W, H, requires_grad=True)
    sf = torch.randn(B, W, R, H)
    sm = torch.ones(B, W, R, dtype=torch.bool)
    valid = torch.ones(B, W, dtype=torch.bool)
    batch = {"chunk_start": [0], "video_id": ["v1"]}
    return batch, valid, h, sf, sm


def test_world_model_loss_skips_missing_future_window_boundary():
    pipe = _import_pipeline_stage1()

    class _FakeIBQ:
        def num_windows(self, vid):
            return 1

        def valid_frame_count(self, vid, window_idx):
            raise AssertionError("missing future window should be skipped before sampling")

        def get(self, vid, window_idx, frame_idx):
            raise AssertionError("missing future window should be skipped before get")

    batch, valid, h, sf, sm = _tiny_world_inputs()
    loss, info = pipe._world_model_loss(
        _TinyWorldModel(), _FakeIBQ(), batch, valid, valid, h, sf, sm,
        1, 16, 32, False,
    )
    assert loss.ndim == 0
    assert info["num_world_windows"] == 0


def test_world_model_loss_uses_existing_future_window_target():
    pipe = _import_pipeline_stage1()

    class _FakeIBQ:
        def __init__(self):
            self.got = []

        def num_windows(self, vid):
            return 2

        def valid_frame_count(self, vid, window_idx):
            return 1

        def get(self, vid, window_idx, frame_idx):
            self.got.append((vid, window_idx, frame_idx))
            return torch.zeros(IBQ_TOKENS_PER_FRAME, dtype=torch.int32)

    ibq = _FakeIBQ()
    batch, valid, h, sf, sm = _tiny_world_inputs()
    _, info = pipe._world_model_loss(
        _TinyWorldModel(), ibq, batch, valid, valid, h, sf, sm,
        1, 16, 32, False,
    )
    assert info["num_world_windows"] == 1
    assert ibq.got == [("v1", 1, 0)]


def test_world_model_loss_propagates_valid_frame_count_index_error():
    pipe = _import_pipeline_stage1()

    class _FakeIBQ:
        def num_windows(self, vid):
            return 2

        def valid_frame_count(self, vid, window_idx):
            raise IndexError("in-range window has invalid metadata")

        def get(self, vid, window_idx, frame_idx):
            raise AssertionError("get should not run")

    batch, valid, h, sf, sm = _tiny_world_inputs()
    with pytest.raises(IndexError, match="invalid metadata"):
        pipe._world_model_loss(
            _TinyWorldModel(), _FakeIBQ(), batch, valid, valid, h, sf, sm,
            1, 16, 32, False,
        )


def test_world_model_loss_propagates_get_index_error_for_existing_window():
    pipe = _import_pipeline_stage1()

    class _FakeIBQ:
        def num_windows(self, vid):
            return 2

        def valid_frame_count(self, vid, window_idx):
            return 1

        def get(self, vid, window_idx, frame_idx):
            raise IndexError("in-range window get failed")

    batch, valid, h, sf, sm = _tiny_world_inputs()
    with pytest.raises(IndexError, match="get failed"):
        pipe._world_model_loss(
            _TinyWorldModel(), _FakeIBQ(), batch, valid, valid, h, sf, sm,
            1, 16, 32, False,
        )


def test_future_ibq_target_samples_only_valid_complete_window():
    pipe = _import_pipeline_stage1()
    calls = []

    class _FakeIBQ:
        def valid_frame_count(self, vid, window_idx):
            return 16

        def get(self, vid, window_idx, frame_idx):
            assert 0 <= frame_idx <= 15
            return torch.zeros(IBQ_TOKENS_PER_FRAME, dtype=torch.int32)

    def _randint(lo, hi):
        calls.append((lo, hi))
        return hi

    old_randint = pipe.random.randint
    try:
        pipe.random.randint = _randint
        tgt = pipe._sample_future_ibq_target(
            _FakeIBQ(), "v1", 0, expected_tokens_per_frame=IBQ_TOKENS_PER_FRAME)
    finally:
        pipe.random.randint = old_randint
    assert calls == [(0, 15)]
    assert tgt.shape == (IBQ_TOKENS_PER_FRAME,)
    print("future IBQ target OK: complete window samples full frame range")


def test_future_ibq_target_samples_only_valid_partial_window():
    pipe = _import_pipeline_stage1()
    calls = []

    class _FakeIBQ:
        def valid_frame_count(self, vid, window_idx):
            return 5

        def get(self, vid, window_idx, frame_idx):
            assert 0 <= frame_idx <= 4
            return torch.zeros(IBQ_TOKENS_PER_FRAME, dtype=torch.int32)

    def _randint(lo, hi):
        calls.append((lo, hi))
        return hi

    old_randint = pipe.random.randint
    try:
        pipe.random.randint = _randint
        pipe._sample_future_ibq_target(
            _FakeIBQ(), "v1", 1, expected_tokens_per_frame=IBQ_TOKENS_PER_FRAME)
    finally:
        pipe.random.randint = old_randint
    assert calls == [(0, 4)]
    print("future IBQ target OK: partial window excludes padded frames")


def test_future_ibq_target_samples_only_single_valid_frame():
    pipe = _import_pipeline_stage1()
    calls = []

    class _FakeIBQ:
        def valid_frame_count(self, vid, window_idx):
            return 1

        def get(self, vid, window_idx, frame_idx):
            assert frame_idx == 0
            return torch.zeros(IBQ_TOKENS_PER_FRAME, dtype=torch.int32)

    def _randint(lo, hi):
        calls.append((lo, hi))
        return hi

    old_randint = pipe.random.randint
    try:
        pipe.random.randint = _randint
        pipe._sample_future_ibq_target(
            _FakeIBQ(), "v1", 1, expected_tokens_per_frame=IBQ_TOKENS_PER_FRAME)
    finally:
        pipe.random.randint = old_randint
    assert calls == [(0, 0)]
    print("future IBQ target OK: single-frame window samples frame 0")


def test_grid_shape_assert():
    assert IBQ_GRID_ROWS * IBQ_GRID_COLS == IBQ_TOKENS_PER_FRAME
    print("test 8 OK: grid shape consistent")


def _make_streaming_model(world_include_decoder=True):
    pipe = _import_pipeline_stage1()

    class _FakeViTForwarder(nn.Module):
        def __init__(self, visual, reducer):
            super().__init__()
            self.visual = visual
            self.reducer = reducer

    old_ssm = pipe.SSMBlock
    old_vit = pipe.ViTForwarder
    try:
        pipe.SSMBlock = _FakeSSM
        pipe.ViTForwarder = _FakeViTForwarder
        model = pipe.StreamingVADGenerationModel(
            _FakeQwen(),
            d_ssm=256,
            llm_hidden=16,
            world_include_decoder=world_include_decoder,
        )
    finally:
        pipe.SSMBlock = old_ssm
        pipe.ViTForwarder = old_vit
    return model


def test_zero_init_modulation_matches_old_formula():
    torch.manual_seed(0)
    model = _make_streaming_model()
    window_batch = torch.randn(1, 3, 16)
    valid = torch.ones(1, 3, dtype=torch.bool)
    state, _, ssm_out, h_internal, _ = model.encode_window_features(
        window_batch, valid, ["v0"], {}, training=True, return_internal=True,
    )
    alpha = torch.sigmoid(model.alpha_logit)
    old_formula = window_batch + alpha * model.adapter(ssm_out)
    assert torch.allclose(state, old_formula, atol=1e-6)
    gamma = torch.tanh(model.temporal_modulator(model.world_branch.temporal_proj(h_internal)))
    assert torch.count_nonzero(gamma).item() == 0
    print("test 9 OK: zero-init temporal modulation is exactly baseline-equivalent")


def test_temporal_modulator_receives_anomaly_gradient():
    torch.manual_seed(0)
    model = _make_streaming_model()
    window_batch = torch.randn(1, 3, 16)
    valid = torch.ones(1, 3, dtype=torch.bool)
    state, *_ = model.encode_window_features(
        window_batch, valid, ["v0"], {}, training=True, return_internal=True,
    )
    state.sum().backward()
    assert model.temporal_modulator[-1].weight.grad is not None
    assert model.temporal_modulator[-1].weight.grad.abs().sum() > 0
    print("test 10 OK: anomaly path gives temporal_modulator gradients")


def test_stage_c_modulation_path_reaches_temporal_proj_and_ssm():
    torch.manual_seed(0)
    model = _make_streaming_model()
    with torch.no_grad():
        model.temporal_modulator[-1].weight.fill_(0.01)
    window_batch = torch.randn(1, 3, 16)
    valid = torch.ones(1, 3, dtype=torch.bool)
    state, _, ssm_out, _, _ = model.encode_window_features(
        window_batch, valid, ["v0"], {}, training=True, return_internal=True,
    )
    alpha = torch.sigmoid(model.alpha_logit)
    old_formula = window_batch + alpha * model.adapter(ssm_out)
    modulation_only = (state - old_formula.detach()).sum()
    modulation_only.backward()
    assert model.world_branch.temporal_proj[0].weight.grad is not None
    assert model.world_branch.temporal_proj[0].weight.grad.abs().sum() > 0
    assert model.ssm.in_proj[0].weight.grad is not None
    assert model.ssm.in_proj[0].weight.grad.abs().sum() > 0
    print("test 11 OK: Stage C modulation path reaches temporal_proj and SSM")


def test_inference_model_runs_without_ibq_decoder():
    torch.manual_seed(0)
    model = _make_streaming_model(world_include_decoder=False)
    assert not hasattr(model.world_branch, "decoder")
    assert not hasattr(model.world_branch, "visual_proj")
    assert model.ibq_codebook.numel() == 0
    window_batch = torch.randn(1, 3, 16)
    valid = torch.ones(1, 3, dtype=torch.bool)
    state, _, _, _ = model.encode_window_features(
        window_batch, valid, ["v0"], {}, training=False, return_internal=False,
    )
    assert state.shape == (1, 3, 16)
    print("test 12 OK: inference model encodes without IBQ decoder/codebook/visual_proj")


def test_legacy_checkpoint_missing_modulator_keeps_zero_init():
    torch.manual_seed(0)
    model = _make_streaming_model()
    pipe = _import_pipeline_stage1()
    ckpt = {"world_branch": model.world_branch.state_dict()}
    pipe._load_temporal_conditioning(model, ckpt)
    assert torch.count_nonzero(model.temporal_modulator[-1].weight).item() == 0
    assert torch.count_nonzero(model.temporal_modulator[-1].bias).item() == 0
    print("test 13 OK: legacy checkpoint without temporal_modulator stays zero-init")


def test_stage_b_warmup_trainability():
    model = _make_streaming_model()
    pipe = _import_pipeline_stage1()
    pipe._set_world_warmup_trainability(model, True)
    assert all(not p.requires_grad for p in model.ssm.parameters())
    assert all(not p.requires_grad for p in model.adapter.parameters())
    assert all(not p.requires_grad for p in model.temporal_modulator.parameters())
    assert all(p.requires_grad for p in model.world_branch.parameters())
    assert all(p.requires_grad for p in model.world_branch.temporal_proj.parameters())
    pipe._set_world_warmup_trainability(model, False)
    assert all(p.requires_grad for p in model.ssm.parameters())
    assert all(p.requires_grad for p in model.adapter.parameters())
    assert all(p.requires_grad for p in model.temporal_modulator.parameters())
    print("test 14 OK: Stage B warmup freezes main path and keeps world_branch trainable")


if __name__ == "__main__":
    test_decoder_produces_per_position_ce()
    test_temporal_proj_structure()
    test_causal_mask_is_upper_triangular()
    test_chunked_ce_equals_full_ce()
    test_zero_temporal_token_is_truly_zero()
    test_joint_gradient_reaches_h_through_temporal_proj()
    test_warmup_detach_blocks_gradient()
    test_world_model_loss_skips_missing_future_window_boundary()
    test_world_model_loss_uses_existing_future_window_target()
    test_world_model_loss_propagates_valid_frame_count_index_error()
    test_world_model_loss_propagates_get_index_error_for_existing_window()
    test_future_ibq_target_samples_only_valid_complete_window()
    test_future_ibq_target_samples_only_valid_partial_window()
    test_future_ibq_target_samples_only_single_valid_frame()
    test_grid_shape_assert()
    test_zero_init_modulation_matches_old_formula()
    test_temporal_modulator_receives_anomaly_gradient()
    test_stage_c_modulation_path_reaches_temporal_proj_and_ssm()
    test_inference_model_runs_without_ibq_decoder()
    test_legacy_checkpoint_missing_modulator_keeps_zero_init()
    test_stage_b_warmup_trainability()
    print("ALL WORLD-MODEL SMOKE TESTS PASSED")
