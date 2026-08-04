import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from hivau_sampler import VideoPairSampler
from mil_utils import (
    abnormal_language_loss,
    abnormal_sparsity_loss,
    anomaly_logits_from_nll,
    cycle_pairs,
    mil_ranking_loss,
    normal_language_loss,
    select_global_max,
)


def test_anomaly_score_direction_positive_for_abnormal():
    normal_nll = torch.tensor([2.0])
    abnormal_nll = torch.tensor([0.2])
    score = anomaly_logits_from_nll(normal_nll, abnormal_nll)
    assert torch.allclose(score, torch.tensor([1.8]))
    assert score.item() > 0


def test_ranking_loss_values():
    ok = mil_ranking_loss(torch.tensor(1.2), torch.tensor(0.3), margin=0.5)
    bad = mil_ranking_loss(torch.tensor(0.6), torch.tensor(0.3), margin=0.5)
    assert ok.item() == 0.0
    assert bad.item() > 0.0


def test_global_max_across_chunks():
    idx, value = select_global_max([
        (0, torch.tensor([0.1, 0.4])),
        (2, torch.tensor([0.3, 0.9])),
    ])
    assert idx == 3
    assert torch.allclose(value, torch.tensor(0.9))


def test_normal_language_loss_uses_all_valid_windows_only():
    nll = torch.tensor([0.2, 0.4, 100.0])
    valid = torch.tensor([True, True, False])
    assert torch.allclose(normal_language_loss(nll, valid), torch.tensor(0.3))


def test_abnormal_language_loss_uses_only_global_max_window():
    abnormal_nll = torch.tensor([0.7, 0.2, 0.9])
    global_indices = torch.tensor([4, 5, 6])
    loss = abnormal_language_loss(abnormal_nll, 5, global_indices)
    assert torch.allclose(loss, torch.tensor(0.2))


def test_sparsity_loss_only_valid_abnormal_probs():
    scores = torch.tensor([0.0, 2.0, -10.0])
    valid = torch.tensor([True, True, False])
    expected = torch.sigmoid(scores[:2]).mean()
    assert torch.allclose(abnormal_sparsity_loss(scores, valid), expected)


def test_video_pair_sampler_preserves_chunk_order_and_cycles_smaller_class():
    samples = [
        {"video_id": "n1", "chunk_start": 1, "chunk_end": 2, "clip_bin": [0]},
        {"video_id": "n1", "chunk_start": 0, "chunk_end": 1, "clip_bin": [0]},
        {"video_id": "a1", "chunk_start": 0, "chunk_end": 1, "clip_bin": [1]},
        {"video_id": "a2", "chunk_start": 0, "chunk_end": 1, "clip_bin": [1]},
    ]
    sampler = VideoPairSampler(samples, shuffle=False)
    pairs = list(sampler.iter_epoch(0))
    assert len(pairs) == 2
    n_vid, n_indices, _, _ = pairs[0]
    assert n_vid == "n1"
    assert n_indices == [1, 0]


class _StatefulSSM:
    def __init__(self):
        self.seen_prev = []

    def forward_chunk(self, x, state=None):
        self.seen_prev.append(state)
        new_state = {"count": 1 if state is None else state["count"] + 1}
        return x, new_state


def test_cross_chunk_state_continuity_and_video_end_clear():
    ssm = _StatefulSSM()
    cache = {}
    vid = "v1"
    _, cache[vid] = ssm.forward_chunk(torch.tensor([[1.0]]), cache.get(vid))
    _, cache[vid] = ssm.forward_chunk(torch.tensor([[2.0]]), cache.get(vid))
    cache.pop(vid, None)
    assert ssm.seen_prev[0] is None
    assert ssm.seen_prev[1] == {"count": 1}
    assert vid not in cache


class _ResidualModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(2, 2, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(-2.1972246))
        nn.init.eye_(self.adapter.weight)

    def forward(self, x, ssm_out):
        return x + torch.sigmoid(self.alpha_logit) * self.adapter(ssm_out)


def test_residual_formula_and_alpha_gradient():
    module = _ResidualModule()
    x = torch.tensor([[1.0, 2.0]])
    ssm_out = torch.tensor([[3.0, 4.0]])
    z = module(x, ssm_out)
    expected = x + torch.sigmoid(module.alpha_logit) * ssm_out
    assert torch.allclose(z, expected)
    z.sum().backward()
    assert module.alpha_logit.grad is not None
    assert module.alpha_logit.grad.abs().item() > 0


def test_checkpoint_round_trip_components_and_mil_hparams():
    ssm = nn.Linear(2, 2)
    adapter = nn.Linear(2, 2)
    alpha_logit = torch.tensor(-2.1972246)
    lora = {"lora_A.weight": torch.randn(2, 2)}
    hparams = {
        "objective": "mil_rank",
        "lambda_normal": 1.0,
        "lambda_abnormal": 1.0,
        "lambda_rank": 1.0,
        "lambda_sparse": 1e-3,
        "mil_margin": 0.5,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "train_state.pt"
        torch.save({
            "ssm": ssm.state_dict(),
            "adapter": adapter.state_dict(),
            "alpha_logit": alpha_logit,
            "lora": lora,
            **hparams,
        }, path)
        loaded = torch.load(path, map_location="cpu")
    for key, value in ssm.state_dict().items():
        assert torch.allclose(value, loaded["ssm"][key])
    for key, value in adapter.state_dict().items():
        assert torch.allclose(value, loaded["adapter"][key])
    assert torch.allclose(alpha_logit, loaded["alpha_logit"])
    assert torch.allclose(lora["lora_A.weight"], loaded["lora"]["lora_A.weight"])
    for key, value in hparams.items():
        assert loaded[key] == value


def test_minimal_mil_training_smoke_step_cpu():
    normal_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    abnormal_scores = torch.tensor([0.2, 1.1, 0.4, 0.6])
    normal_nll = torch.nn.Parameter(torch.ones(4) * 0.5)
    abnormal_nll_at_max = torch.nn.Parameter(torch.tensor(0.25))
    scale = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([normal_nll, abnormal_nll_at_max, scale], lr=0.01)

    normal_max = (normal_scores * scale).max()
    abnormal_logits = abnormal_scores * scale
    abnormal_max = abnormal_logits.max()
    loss = (
        normal_nll.mean()
        + abnormal_nll_at_max
        + mil_ranking_loss(abnormal_max, normal_max, margin=0.5)
        + 1e-3 * torch.sigmoid(abnormal_logits).mean()
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert normal_nll.grad is not None
    assert abnormal_nll_at_max.grad is not None
    assert scale.grad is not None
