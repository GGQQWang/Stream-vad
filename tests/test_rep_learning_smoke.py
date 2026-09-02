"""CPU smoke tests for the Progressive Compression Predictive
Representation Learning branch (no full training, no Qwen, no mamba).

Run on the server:
    PYTHONPATH=. python tests/test_rep_learning_smoke.py
"""

import torch
import torch.nn as nn

from rep_learning import (
    FuturePredictor,
    SIGProjector,
    clear_rep_finished_states,
    configure_rep_modules,
    deterministic_rho,
    future_loss,
    init_ema,
    invariance_loss_from_q,
    rep_loss_forward,
    sigreg_epps_pulley,
    update_ema,
)
from spatial import SpatialTokenCompressor


class _FakeSSM(nn.Module):
    """Linear stand-in for SSMBlock.forward_chunk (CPU smoke tests only)."""

    def __init__(self, d_input, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_input, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_input)

    def forward_chunk(self, x, state=None, return_internal=False):
        h = self.in_proj(x)
        prev = state[0] if state else torch.zeros_like(h)
        h = h + prev                              # state carry is observable
        new_state = {0: h[:, -1:].detach()}
        h = self.out_norm(h)
        out = self.out_proj(h)
        if return_internal:
            return out, new_state, h
        return out, new_state


class _FakeModel(nn.Module):
    """Minimal model surface used by rep_loss_forward."""

    def __init__(self, llm_hidden=16, d_ssm=8):
        super().__init__()
        self.llm_hidden = llm_hidden
        self.spatial = SpatialTokenCompressor(reduction_ratio=0.5, k=3)
        self.ssm = _FakeSSM(llm_hidden, d_ssm)
        self.ema_ssm = _FakeSSM(llm_hidden, d_ssm)
        self.future_predictor = FuturePredictor(d_ssm, hidden=16)
        self.sig_projector = SIGProjector(d_ssm, hidden=16)


def _synthetic_batch(B=1, T=6, llm_hidden=16, d_ssm=8, r_max=8):
    window_batch = torch.randn(B, T, llm_hidden)
    spatial_features = torch.randn(B, T, r_max, llm_hidden)
    spatial_mask = torch.ones(B, T, r_max, dtype=torch.bool)
    valid_mask = torch.ones(B, T, dtype=torch.bool)
    batch = {"video_id": [f"v{b}" for b in range(B)], "is_last_chunk": [False] * B}
    return window_batch, spatial_features, spatial_mask, valid_mask, batch


def test_secondary_lof_smaller_rho_keeps_fewer_tokens():
    torch.manual_seed(0)
    comp = SpatialTokenCompressor(reduction_ratio=0.5, k=8)
    x = torch.randn(64, 16)
    c_hi, _ = comp(x, 0.9)
    c_lo, _ = comp(x, 0.4)
    c_def, _ = comp(x)
    assert c_lo.shape[0] <= c_hi.shape[0]
    assert c_hi.shape[0] == int(64 * 0.9)
    assert c_lo.shape[0] == int(64 * 0.4)
    # ratio=None keeps the original configured behaviour
    assert c_def.shape[0] == int(64 * 0.5)
    print("test 1 OK: secondary LOF: smaller rho -> fewer/equal tokens")


def test_deterministic_rho():
    r_a = deterministic_rho(42, 0, "vidA", 0.75, 0.95, salt="l1")
    r_b = deterministic_rho(42, 0, "vidA", 0.75, 0.95, salt="l1")
    r_c = deterministic_rho(42, 1, "vidA", 0.75, 0.95, salt="l1")
    r_d = deterministic_rho(42, 0, "vidA", 0.75, 0.95, salt="l2")
    assert r_a == r_b, "same video+epoch must be identical"
    assert r_a != r_c, "different epoch must differ"
    assert r_a != r_d, "different salt must decorrelate views"
    assert 0.75 <= r_a <= 0.95
    r2 = deterministic_rho(7, 3, "vidB", 0.50, 0.75)
    assert 0.50 <= r2 <= 0.75
    print("test 2 OK: deterministic per-(epoch, video) ratio in range")


def test_future_predictor_shapes_and_horizons():
    torch.manual_seed(0)
    P = FuturePredictor(8, hidden=16)
    assert not torch.allclose(P.e1, P.e2), "horizon embeddings must differ"
    h = torch.randn(2, 3, 8, requires_grad=True)
    p1, p2 = P(h)
    assert p1.shape == (2, 3, 8) and p2.shape == (2, 3, 8)
    # t+2 must NOT depend on pred_t1 (parallel prediction)
    g = torch.autograd.grad(p2.sum(), p1, allow_unused=True, retain_graph=True)[0]
    assert g is None, "pred_t2 must not depend on pred_t1"
    gh = torch.autograd.grad(p1.sum() + p2.sum(), h, retain_graph=True)[0]
    assert gh is not None and gh.abs().sum() > 0
    print("test 3 OK: predictor shapes, distinct horizons, no t1->t2 chain")


def test_warmup_future_loss_trains_only_predictor():
    torch.manual_seed(0)
    P = FuturePredictor(8, hidden=16)
    h_src = nn.Parameter(torch.randn(2, 4, 8))          # stands for SSM output
    h_bar = torch.randn(2, 4, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    w = torch.ones(2, 3)
    loss, info = future_loss(
        P, [h_src.detach(), h_src.detach(), h_src.detach()],
        h_bar, valid, w, horizon2_weight=0.5, detach_inputs=True,
    )
    assert info["n_anchors"] == 2 * 2
    loss.backward()
    assert h_src.grad is None or h_src.grad.abs().sum() == 0, (
        "warmup: future loss must not touch the SSM"
    )
    assert P.mlp[0].weight.grad is not None and P.mlp[0].weight.grad.abs().sum() > 0
    print("test 4 OK: warmup detach -> predictor grad != 0, SSM grad == 0")


def test_formal_future_and_inv_losses_reach_ssm():
    torch.manual_seed(0)
    P = FuturePredictor(8, hidden=16)
    h_src = nn.Parameter(torch.randn(2, 4, 8))
    h_bar = torch.randn(2, 4, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    w = torch.ones(2, 3)
    loss, _ = future_loss(
        P, [h_src, h_src, h_src], h_bar, valid, w, 0.5, detach_inputs=False,
    )
    loss.backward()
    assert h_src.grad is not None and h_src.grad.abs().sum() > 0, (
        "formal: future loss must reach the SSM"
    )
    h_src.grad = None
    G = SIGProjector(8, hidden=16)
    h_l1 = nn.Parameter(torch.randn(2, 4, 8))
    h_l2 = nn.Parameter(torch.randn(2, 4, 8))
    loss_inv = invariance_loss_from_q(G(h_src), G(h_l1), G(h_l2), valid)
    loss_inv.backward()
    assert h_src.grad is not None and h_src.grad.abs().sum() > 0, (
        "formal: invariance loss must reach the global h"
    )
    print("test 5 OK: formal phase -> future and invariance gradients reach the SSM")


def test_sigreg_finite_and_zero_fallback():
    torch.manual_seed(0)
    q = torch.randn(3, 12, 8, requires_grad=True)
    s = sigreg_epps_pulley(q, num_proj=64, knots=17)
    assert torch.isfinite(s) and s.item() >= 0
    s.backward()
    assert q.grad is not None
    q2 = torch.randn(1, 1, 8, requires_grad=True)      # V*N < 2 -> fallback
    s2 = sigreg_epps_pulley(q2, num_proj=64, knots=17)
    assert s2.item() == 0.0 and s2.requires_grad
    s2.backward()
    print("test 6 OK: SIGReg finite; small-sample fallback is graph-connected zero")


def test_ema_frozen_not_in_optimizer_updates_on_step():
    torch.manual_seed(0)
    online = _FakeSSM(4, 8)
    ema = _FakeSSM(4, 8)
    init_ema(ema, online)
    for pe, po in zip(ema.parameters(), online.parameters()):
        assert torch.equal(pe.data, po.data), "init_ema must copy the online SSM"
    opt = torch.optim.AdamW(online.parameters())
    opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    assert all(not p.requires_grad for p in ema.parameters())
    assert not any(id(p) in opt_ids for p in ema.parameters()), (
        "EMA must not be inside the optimizer"
    )
    with torch.no_grad():
        online.in_proj.weight.data.fill_(2.0)
    update_ema(ema, online, 0.996)                     # ema was 1.0*orig?
    # re-init to a known state for a clean math check
    with torch.no_grad():
        online.in_proj.weight.data.fill_(1.0)
    init_ema(ema, online)
    with torch.no_grad():
        online.in_proj.weight.data.fill_(3.0)
    update_ema(ema, online, 0.996)
    expected = 0.996 * 1.0 + 0.004 * 3.0
    assert torch.allclose(ema.in_proj.weight.data, torch.full_like(
        ema.in_proj.weight.data, expected), atol=1e-6)
    print("test 7 OK: EMA frozen, outside the optimizer, exact momentum math")


def test_baseline_lambda_rep_zero_freezes_rep_modules():
    model = _FakeModel()
    configure_rep_modules(model, 0.0)
    rep_mods = (model.future_predictor, model.sig_projector, model.ema_ssm)
    rep_ids = {id(p) for m in rep_mods for p in m.parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert not (set(map(id, trainable)) & rep_ids), (
        "lambda_rep=0 must freeze all rep modules (baseline optimizer set unchanged)"
    )
    configure_rep_modules(model, 0.1)
    assert all(p.requires_grad for p in model.future_predictor.parameters())
    assert all(p.requires_grad for p in model.sig_projector.parameters())
    assert all(not p.requires_grad for p in model.ema_ssm.parameters()), (
        "EMA must stay frozen even when the rep branch is active"
    )
    print("test 8 OK: baseline freeze + EMA always frozen")


def test_rep_mode_has_no_ibq_dependency():
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "rep_learning.py"
    text = src.read_text().lower()
    assert "ibq" not in text, "the rep branch must not depend on the IBQ cache"
    pipe = (Path(__file__).resolve().parent.parent / "pipeline_stage1.py").read_text()
    marker = "--- Progressive Compression Predictive Representation"
    tail = pipe[pipe.index(marker):pipe.index("total_loss = raw_total_loss", pipe.index(marker))]
    assert "ibq" not in tail.lower(), "the rep loss block must not reference IBQ"
    print("test 9 OK: rep mode is completely IBQ-free")


def test_rep_loss_forward_integration():
    """End-to-end through rep_loss_forward with fake modules: caches stay
    independent, warmup detaches, formal phase trains SSM + predictor +
    projector, EMA never receives grad, rho is stable within an epoch."""
    torch.manual_seed(0)
    window_batch, sf, sm, valid, batch = _synthetic_batch(T=6)

    def make_model():
        return _FakeModel(llm_hidden=16, d_ssm=8)

    # ---- warmup pass ----
    model = make_model()
    _, _, h_internal = model.ssm.forward_chunk(
        window_batch, state=None, return_internal=True,
    )
    caches = {"local1": {}, "local2": {}, "ema": {}}
    loss, info = rep_loss_forward(
        model, window_batch, sf, sm, valid, h_internal, batch,
        epoch=0, seed=42,
        rho_l1_range=(0.75, 0.95), rho_l2_range=(0.50, 0.75),
        horizon2_weight=0.5, sig_weight=0.02,
        sig_num_proj=64, sig_knots=17,
        detach_inputs=True, rep_caches=caches,
    )
    assert torch.isfinite(loss)
    assert info["loss_inv"] == 0.0 and info["loss_sigreg"] == 0.0
    loss.backward()
    assert model.ssm.in_proj.weight.grad is None or model.ssm.in_proj.weight.grad.abs().sum() == 0
    assert model.future_predictor.mlp[0].weight.grad is not None
    assert model.sig_projector.mlp[0].weight.grad is None, "projector must not train during warmup"
    for key in ("local1", "local2", "ema"):
        assert "v0" in caches[key], f"cache {key} must carry the video state"
    # warmup rho for later comparison
    rho_warmup = info["rho_local1"]
    assert 0.75 <= rho_warmup <= 0.95

    # ---- formal pass (fresh model, same epoch -> same rho) ----
    model = make_model()
    _, _, h_internal = model.ssm.forward_chunk(
        window_batch, state=None, return_internal=True,
    )
    caches = {"local1": {}, "local2": {}, "ema": {}}
    loss, info = rep_loss_forward(
        model, window_batch, sf, sm, valid, h_internal, batch,
        epoch=0, seed=42,
        rho_l1_range=(0.75, 0.95), rho_l2_range=(0.50, 0.75),
        horizon2_weight=0.5, sig_weight=0.02,
        sig_num_proj=64, sig_knots=17,
        detach_inputs=False, rep_caches=caches,
    )
    assert info["rho_local1"] == rho_warmup, "same video+epoch must reuse the same rho"
    assert info["n_anchors"] == 4                     # T=6 -> t=0..3
    assert torch.isfinite(info["loss_sigreg"]) and info["loss_sigreg"] >= 0
    loss.backward()
    assert model.ssm.in_proj.weight.grad is not None and model.ssm.in_proj.weight.grad.abs().sum() > 0
    assert model.future_predictor.mlp[0].weight.grad is not None
    assert model.sig_projector.mlp[0].weight.grad is not None
    assert all(p.grad is None for p in model.ema_ssm.parameters()), (
        "EMA must never receive gradients"
    )
    # caches are independent dicts with independent state tensors
    assert caches["local1"] is not caches["local2"] and caches["local2"] is not caches["ema"]
    assert not torch.allclose(
        caches["local1"]["v0"][0], caches["local2"]["v0"][0],
    ), "local1/local2 must carry independent states"

    # ---- a different epoch changes the ratios ----
    caches = {"local1": {}, "local2": {}, "ema": {}}
    _, _, h_internal = model.ssm.forward_chunk(
        window_batch, state=None, return_internal=True,
    )
    _, info_ep1 = rep_loss_forward(
        model, window_batch, sf, sm, valid, h_internal, batch,
        epoch=1, seed=42,
        rho_l1_range=(0.75, 0.95), rho_l2_range=(0.50, 0.75),
        horizon2_weight=0.5, sig_weight=0.02,
        sig_num_proj=64, sig_knots=17,
        detach_inputs=False, rep_caches=caches,
    )
    assert info_ep1["rho_local1"] != rho_warmup, "different epoch must draw a different ratio"

    # ---- finished-video state clearing ----
    batch_done = {"video_id": ["v0"], "is_last_chunk": [True]}
    clear_rep_finished_states(batch_done, caches)
    assert "v0" not in caches["local1"] and "v0" not in caches["ema"]
    print("test 10 OK: rep_loss_forward integration (caches, warmup, formal, EMA, rho, clearing)")


if __name__ == "__main__":
    test_secondary_lof_smaller_rho_keeps_fewer_tokens()
    test_deterministic_rho()
    test_future_predictor_shapes_and_horizons()
    test_warmup_future_loss_trains_only_predictor()
    test_formal_future_and_inv_losses_reach_ssm()
    test_sigreg_finite_and_zero_fallback()
    test_ema_frozen_not_in_optimizer_updates_on_step()
    test_baseline_lambda_rep_zero_freezes_rep_modules()
    test_rep_mode_has_no_ibq_dependency()
    test_rep_loss_forward_integration()
    print("ALL REP-LEARNING SMOKE TESTS PASSED")
