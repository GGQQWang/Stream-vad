"""Training-only Progressive Compression Predictive Representation Learning.

The branch turns the SSM internal state ``h_internal [256]`` into a
robust, causal, future-predictive representation WITHOUT touching the
anomaly-score / summary main tasks:

  - Global view: the existing standard LOF compression (v3 cache), reused
    from the main path — the global SSM is never re-run.
  - Local views: a SECONDARY LOF pass over the v3 ``spatial_features``
    with per-(epoch, video) deterministic ratios ``rho_l1``/``rho_l2``,
    mean-pooled to ``x_l1``/``x_l2`` and sent through the SAME online
    SSM with independent runtime state caches.
  - EMA future teacher: ``ema_ssm`` (EMA copy of ``model.ssm``,
    requires_grad=False, eval) consumes only the global compressed
    features and provides stop-grad future targets ``h_bar_g[t+1]``,
    ``h_bar_g[t+2]``.
  - Future predictor: ``P(h_t + e_k)`` with learned per-horizon
    embeddings ``e1``/``e2`` (parallel, t+2 never depends on pred_t1).
  - Invariance: symmetric MSE of ``G(h_l1)``/``G(h_l2)`` against
    ``G(h_g)`` (single online encoder, global NOT stop-graded).
  - SIGReg: sliced Epps-Pulley Gaussianity statistic over
    ``stack([q_g, q_l1, q_l2])`` (LeVJEPA-style; no VICReg substitute).

Warmup (first ``--rep-warmup-steps`` optimizer updates): the future
predictor sees ``h.detach()`` (predictor trains, SSM untouched) and
invariance/SIGReg are zeroed.  Afterwards all three terms train jointly.

Nothing here runs at inference time; ``infer_stage1_ucf.py`` is
unaffected.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# deterministic progressive-compression ratios
# ---------------------------------------------------------------------------

def deterministic_rho(
    seed: int,
    epoch: int,
    video_id: str,
    lo: float,
    hi: float,
    salt: str = "",
) -> float:
    """Deterministic per-(epoch, video) ratio in ``[lo, hi]``.

    Same video in the same epoch always draws the same value (across
    chunk boundaries and resume); different epochs draw different
    values.  ``salt`` separates the two local views so they are
    decorrelated.
    """
    key = f"{seed}:{epoch}:{video_id}:{salt}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    u = int.from_bytes(digest, "little") / float(2 ** 64)
    return lo + (hi - lo) * u


# ---------------------------------------------------------------------------
# training-only modules
# ---------------------------------------------------------------------------

class FuturePredictor(nn.Module):
    """Shared future predictor over one current hidden ``h_t``.

    ``pred_t1 = P(h + e1)`` and ``pred_t2 = P(h + e2)`` are computed in
    PARALLEL — ``pred_t2`` never depends on ``pred_t1``, and no delta
    residual is predicted.
    """

    def __init__(self, d: int = 256, hidden: int = 512):
        super().__init__()
        self.e1 = nn.Parameter(torch.zeros(1, d))
        # small noise breaks the t1/t2 symmetry from step 0
        self.e2 = nn.Parameter(torch.randn(1, d) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_t1 = self.mlp(h + self.e1)
        pred_t2 = self.mlp(h + self.e2)
        return pred_t1, pred_t2


class SIGProjector(nn.Module):
    """LeVJEPA-style buffer projector (no LN/BN/dropout/L2 norm)."""

    def __init__(self, d: int = 256, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


def configure_rep_modules(model, lambda_rep: float) -> bool:
    """Freeze/train the training-only rep modules.

    ``lambda_rep=0``: future_predictor / sig_projector frozen so the
    baseline optimizer parameter set and training path are unchanged.
    The EMA SSM is ALWAYS requires_grad=False and in eval mode.
    Returns True when the rep branch is active.
    """
    active = bool(lambda_rep > 0)
    for m in (model.future_predictor, model.sig_projector):
        for p in m.parameters():
            p.requires_grad = active
    for p in model.ema_ssm.parameters():
        p.requires_grad = False
    model.ema_ssm.eval()
    return active


@torch.no_grad()
def init_ema(ema_module: nn.Module, online_module: nn.Module) -> None:
    """``ema_module`` <- ``online_module`` (weights AND every buffer)."""
    ema_module.load_state_dict(online_module.state_dict())
    for eb, ob in zip(ema_module.buffers(), online_module.buffers()):
        eb.data.copy_(ob.data)


@torch.no_grad()
def update_ema(ema_module: nn.Module, online_module: nn.Module,
               momentum: float) -> None:
    """``ema = momentum * ema + (1 - momentum) * online`` in float32.

    Called once per REAL ``optimizer.step()`` — never on a grad-accum
    micro-batch.  Non-floating buffers are copied in place.
    """
    for ep, op in zip(ema_module.parameters(), online_module.parameters()):
        ef = ep.data.float()
        of = op.data.float()
        ef.mul_(momentum).add_(of, alpha=1.0 - momentum)
        ep.data.copy_(ef)
    for eb, ob in zip(ema_module.buffers(), online_module.buffers()):
        if not eb.dtype.is_floating_point:
            eb.data.copy_(ob.data)
            continue
        ef = eb.data.float()
        of = ob.data.float()
        ef.mul_(momentum).add_(of, alpha=1.0 - momentum)
        eb.data.copy_(ef)


def clear_rep_finished_states(batch: dict, rep_caches: Dict[str, dict]) -> None:
    """Drop per-video state of finished videos from every rep cache."""
    for vid, is_last in zip(batch["video_id"], batch["is_last_chunk"]):
        if is_last:
            for cache in rep_caches.values():
                cache.pop(vid, None)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def future_loss(
    P: FuturePredictor,
    h_views: List[torch.Tensor],      # [B, T, d] for (global, local1, local2)
    h_bar: torch.Tensor,              # [B, T, d] EMA teacher targets
    valid_mask: torch.Tensor,         # [B, T] bool
    view_weights: torch.Tensor,       # [B, 3] float (1.0, rho_l1, rho_l2)
    horizon2_weight: float,
    detach_inputs: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Future prediction MSE over chunk-internal anchors.

    An anchor is a window ``t`` whose ``t+1`` AND ``t+2`` are both valid
    within the same chunk (no cross-chunk targets).  Targets are
    stop-grad EMA hiddens; ``detach_inputs`` additionally detaches the
    predictor INPUT so only the predictor receives gradients (warmup).
    """
    B, T, _ = h_bar.shape
    valid = valid_mask.bool()
    anchor = (valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]).to(h_bar.device)  # [B, T-2]
    tgt1 = h_bar[:, 1:-1].detach()                            # per anchor t: t+1
    tgt2 = h_bar[:, 2:].detach()                              # per anchor t: t+2

    if not bool(anchor.any()):
        # no future pair in this chunk: graph-connected zero through the
        # predictor so backward() still works
        h_in = h_views[0].detach() if detach_inputs else h_views[0]
        z = P(h_in.sum(dim=(0, 1)))[0].sum() * 0.0
        return z, {
            "loss_future": 0.0, "loss_t1": 0.0, "loss_t2": 0.0,
            "future_global": 0.0, "future_local1": 0.0, "future_local2": 0.0,
            "n_anchors": 0,
        }

    per_view: List[torch.Tensor] = []
    mse_t1_views: List[torch.Tensor] = []
    mse_t2_views: List[torch.Tensor] = []
    for h_v in h_views:
        h_in = h_v.detach() if detach_inputs else h_v
        p1, p2 = P(h_in)                                       # [B, T, d]
        mse1 = (p1[:, :-2] - tgt1).pow(2).mean(dim=-1)         # [B, T-2]
        mse2 = (p2[:, :-2] - tgt2).pow(2).mean(dim=-1)
        mse1_m = (mse1 * anchor).sum() / anchor.sum().clamp_min(1)
        mse2_m = (mse2 * anchor).sum() / anchor.sum().clamp_min(1)
        per_view.append(mse1_m + horizon2_weight * mse2_m)     # Lf_v
        mse_t1_views.append(mse1_m.detach())
        mse_t2_views.append(mse2_m.detach())

    w = view_weights.to(h_bar.device)                          # [B, 3]
    w_sum = w.sum().clamp_min(1e-8)
    loss = sum(w[:, v].sum() * per_view[v] for v in range(3)) / w_sum

    info = {
        "loss_future": float(loss.detach().item()),
        "loss_t1": float(sum(mse_t1_views).item() / len(mse_t1_views)),
        "loss_t2": float(sum(mse_t2_views).item() / len(mse_t2_views)),
        "future_global": float(per_view[0].detach().item()),
        "future_local1": float(per_view[1].detach().item()),
        "future_local2": float(per_view[2].detach().item()),
        "n_anchors": int(anchor.sum().item()),
    }
    return loss, info


def invariance_loss_from_q(
    q_g: torch.Tensor,               # [B, T, d]
    q_l1: torch.Tensor,
    q_l2: torch.Tensor,
    valid_mask: torch.Tensor,        # [B, T] bool
) -> torch.Tensor:
    """Symmetric single-encoder invariance: 0.5*(MSE(q_l1,q_g)+MSE(q_l2,q_g)).

    The global view is NOT stop-graded.
    """
    valid = valid_mask.bool().to(q_g.device)
    if not bool(valid.any()):
        return q_g.sum() * 0.0
    mse1 = (q_l1 - q_g).pow(2).mean(dim=-1)                    # [B, T]
    mse2 = (q_l2 - q_g).pow(2).mean(dim=-1)
    m1 = (mse1 * valid).sum() / valid.sum().clamp_min(1)
    m2 = (mse2 * valid).sum() / valid.sum().clamp_min(1)
    return 0.5 * (m1 + m2)


def sigreg_epps_pulley(
    q_views: torch.Tensor,           # [V, N, d] stacked projections
    num_proj: int = 1024,
    knots: int = 17,
) -> torch.Tensor:
    """Sliced Epps-Pulley Gaussianity statistic (SIGReg).

    Projects the stacked views onto random unit Gaussian directions,
    then compares the empirical characteristic function of each 1-D
    projection with the standard Gaussian characteristic function
    ``exp(-t^2/2)`` on a fixed knot grid (real AND imaginary parts).

    Forced to float32 internally; ``normalize_by_n=False`` (the raw
    statistic, no sample-size scaling).  Returns a graph-connected zero
    when there are too few samples.
    """
    V, N, d = q_views.shape
    if V * N < 2:
        return q_views.mean() * 0.0
    x = q_views.float()
    device = x.device
    g = torch.randn(num_proj, d, device=device, dtype=torch.float32)
    g = F.normalize(g, dim=-1)                                 # [P, d]
    y = torch.einsum("vnd,pd->vnp", x, g)                      # [V, N, P]
    y_flat = y.reshape(-1, num_proj)                           # [V*N, P]
    mu = y_flat.mean(dim=0, keepdim=True)
    std = y_flat.std(dim=0, keepdim=True).clamp_min(1e-4)
    z = (y_flat - mu) / std                                    # [V*N, P]
    ts = torch.linspace(-2.0, 2.0, knots, device=device, dtype=torch.float32)
    arg = ts.view(knots, 1, 1) * z.t().view(1, num_proj, -1)   # [knots, P, V*N]
    phi_real = arg.cos().mean(dim=-1)                          # [knots, P]
    phi_imag = arg.sin().mean(dim=-1)
    target = torch.exp(-0.5 * ts * ts).view(knots, 1)          # [knots, 1]
    err = (phi_real - target).square() + phi_imag.square()     # [knots, P]
    return err.mean()


# ---------------------------------------------------------------------------
# full-batch forward
# ---------------------------------------------------------------------------

def rep_loss_forward(
    model,
    window_batch: torch.Tensor,      # [B, T, llm_hidden] global x_t (device)
    spatial_features: torch.Tensor,  # [B, T, R_max, llm_hidden] (v3 cache)
    spatial_mask: torch.Tensor,      # [B, T, R_max] bool
    valid_mask_cpu: torch.Tensor,    # [B, T] bool
    h_internal: torch.Tensor,        # [B, T, d_ssm] global SSM internal
    batch: dict,                     # video_id / is_last_chunk
    epoch: int,
    seed: int,
    rho_l1_range: Tuple[float, float],
    rho_l2_range: Tuple[float, float],
    horizon2_weight: float,
    sig_weight: float,
    sig_num_proj: int,
    sig_knots: int,
    detach_inputs: bool,             # warmup: predictor sees detached h
    rep_caches: Dict[str, dict],     # {"local1": .., "local2": .., "ema": ..}
) -> Tuple[torch.Tensor, dict]:
    """One batch of the Progressive Compression Predictive Representation
    loss: ``L_rep = L_future + L_inv + sig_weight * L_sig``."""
    B, T, _ = h_internal.shape
    d_ssm = h_internal.shape[-1]
    ssm_param = next(model.ssm.parameters())
    dev, ssm_dtype = ssm_param.device, ssm_param.dtype
    valid_b, valid_w = valid_mask_cpu.nonzero(as_tuple=True)

    # ---- per-(epoch, video) progressive ratios (fixed within a video) ----
    rho_map: Dict[str, Tuple[float, float]] = {}
    for b in range(B):
        vid = batch["video_id"][b]
        if vid in rho_map:
            continue
        rho_map[vid] = (
            deterministic_rho(seed, epoch, vid, *rho_l1_range, salt="l1"),
            deterministic_rho(seed, epoch, vid, *rho_l2_range, salt="l2"),
        )

    # ---- secondary LOF over the v3 spatial cache (no ViT re-run) ----
    x_l1 = torch.zeros(B, T, model.llm_hidden,
                       device=window_batch.device, dtype=window_batch.dtype)
    x_l2 = torch.zeros_like(x_l1)
    for i in range(len(valid_b)):
        b = int(valid_b[i].item())
        w = int(valid_w[i].item())
        r1, r2 = rho_map[batch["video_id"][b]]
        C = spatial_features[b, w][spatial_mask[b, w].bool()]  # [R, C]
        c_l1, _ = model.spatial(C, r1)
        c_l2, _ = model.spatial(C, r2)
        x_l1[b, w] = c_l1.mean(dim=0)
        x_l2[b, w] = c_l2.mean(dim=0)

    # ---- local views through the SHARED online SSM, separate caches ----
    h_l1 = torch.zeros(B, T, d_ssm, device=dev, dtype=ssm_dtype)
    h_l2 = torch.zeros(B, T, d_ssm, device=dev, dtype=ssm_dtype)
    for key, x_view, h_out in (("local1", x_l1, h_l1), ("local2", x_l2, h_l2)):
        for b in range(B):
            vid = batch["video_id"][b]
            bw = valid_w[valid_b == b]
            if len(bw) == 0:
                continue
            xb = x_view[b:b + 1, bw].to(device=dev, dtype=ssm_dtype)
            prev = rep_caches[key].get(vid)
            if prev is not None:            # TBPTT detach across chunks
                prev = {i: s.detach() for i, s in prev.items()}
            _, new_st, internal = model.ssm.forward_chunk(
                xb, state=prev, return_internal=True,
            )
            h_out[b, bw] = internal.squeeze(0)
            rep_caches[key][vid] = new_st

    # ---- EMA future teacher: global compressed features only, no grad ----
    h_bar = torch.zeros(B, T, d_ssm, device=dev, dtype=ssm_dtype)
    was_training = model.ema_ssm.training
    model.ema_ssm.eval()
    try:
        with torch.no_grad():
            for b in range(B):
                vid = batch["video_id"][b]
                bw = valid_w[valid_b == b]
                if len(bw) == 0:
                    continue
                xb = window_batch[b:b + 1, bw].to(device=dev, dtype=ssm_dtype)
                prev = rep_caches["ema"].get(vid)
                if prev is not None:
                    prev = {i: s.detach() for i, s in prev.items()}
                _, new_st, internal = model.ema_ssm.forward_chunk(
                    xb, state=prev, return_internal=True,
                )
                h_bar[b, bw] = internal.squeeze(0)
                rep_caches["ema"][vid] = new_st
    finally:
        model.ema_ssm.train(was_training)

    # ---- losses ----
    h_g = h_internal.to(device=dev, dtype=ssm_dtype)
    view_weights = torch.ones(B, 3, device=dev, dtype=torch.float32)
    for b in range(B):
        r1, r2 = rho_map[batch["video_id"][b]]
        view_weights[b, 1] = r1
        view_weights[b, 2] = r2

    loss_future, f_info = future_loss(
        model.future_predictor, [h_g, h_l1, h_l2], h_bar,
        valid_mask_cpu, view_weights, horizon2_weight,
        detach_inputs=detach_inputs,
    )

    q_g = None
    if detach_inputs:
        # predictor warmup: invariance / SIGReg disabled; graph-connected
        # zero so backward() through the branch still works
        loss_inv = h_g.sum() * 0.0
        loss_sig = h_g.sum() * 0.0
    else:
        q_g = model.sig_projector(h_g)
        q_l1 = model.sig_projector(h_l1)
        q_l2 = model.sig_projector(h_l2)
        loss_inv = invariance_loss_from_q(q_g, q_l1, q_l2, valid_mask_cpu)
        q_stack = torch.stack([q_g, q_l1, q_l2], dim=0)        # [3, B, T, d]
        loss_sig = sigreg_epps_pulley(
            q_stack[:, valid_mask_cpu.bool()],                 # [3, N, d]
            num_proj=sig_num_proj, knots=sig_knots,
        )

    loss_rep = loss_future + loss_inv + sig_weight * loss_sig

    with torch.no_grad():
        v = valid_mask_cpu.bool()
        if bool(v.any()):
            norm_h_global = float(h_g[v].float().norm(dim=-1).mean().item())
            norm_h_local1 = float(h_l1[v].float().norm(dim=-1).mean().item())
            norm_h_local2 = float(h_l2[v].float().norm(dim=-1).mean().item())
            norm_q_global = (
                float(q_g[v].float().norm(dim=-1).mean().item())
                if q_g is not None else float("nan")
            )
        else:
            norm_h_global = norm_h_local1 = norm_h_local2 = norm_q_global = float("nan")

    info = {
        "loss_future": f_info["loss_future"],
        "loss_future_t1": f_info["loss_t1"],
        "loss_future_t2": f_info["loss_t2"],
        "future_global": f_info["future_global"],
        "future_local1": f_info["future_local1"],
        "future_local2": f_info["future_local2"],
        "loss_inv": float(loss_inv.detach().item()),
        "loss_sigreg": float(loss_sig.detach().item()),
        "rho_local1": float(view_weights[:, 1].mean().item()),
        "rho_local2": float(view_weights[:, 2].mean().item()),
        "n_anchors": f_info["n_anchors"],
        "norm_h_global": norm_h_global,
        "norm_h_local1": norm_h_local1,
        "norm_h_local2": norm_h_local2,
        "norm_q_global": norm_q_global,
    }
    return loss_rep, info
