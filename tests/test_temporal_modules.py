import pytest
import torch
import torch.nn as nn

from stage_ap import evaluate_video_stage_ap
from temporal_modules import (
    TEMPORAL_MODELS,
    build_temporal_module,
    checkpoint_temporal_config,
)


class _LegacySSM(nn.Module):
    """Tiny stand-in proving the factory adds no wrapper or state-dict prefix."""

    def __init__(self, d_input, d_model, n_layers, llm_hidden):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(d_input, d_model), nn.LayerNorm(d_model))
        self.out_proj = nn.Linear(d_model, llm_hidden)

    def forward_chunk(self, x, state=None, return_internal=False):
        internal = self.in_proj(x)
        output = self.out_proj(internal)
        memory = {0: internal[:, -1]}
        if return_internal:
            return output, memory, internal
        return output, memory

    def step(self, x_t, memory=None):
        _, new_memory, internal = self.forward_chunk(
            x_t.unsqueeze(1), state=memory, return_internal=True,
        )
        return internal[:, 0], new_memory


def _build(name, history=4):
    return build_temporal_module(
        name,
        d_input=8,
        d_state=8,
        output_dim=4,
        history_length=history,
        ssm_cls=_LegacySSM if name == "ssm" else None,
    )


@pytest.mark.parametrize("name", TEMPORAL_MODELS)
def test_build_temporal_modules_have_common_output_shapes(name):
    torch.manual_seed(1)
    module = _build(name).eval()
    output, memory, state = module.forward_chunk(
        torch.randn(3, 5, 8), return_internal=True,
    )
    assert output.shape == (3, 5, 4)
    assert state.shape == (3, 5, 8)
    assert memory is not None


@pytest.mark.parametrize("name", TEMPORAL_MODELS)
def test_temporal_step_returns_unified_causal_state_shape(name):
    module = _build(name).eval()
    state, memory = module.step(torch.randn(3, 8))
    assert state.shape == (3, 8)
    assert memory is not None


def test_lstm_causal_step_and_reset():
    torch.manual_seed(2)
    module = _build("lstm").eval()
    x = torch.randn(2, 8)
    first, memory = module.step(x)
    second, _ = module.step(x, memory)
    reset_first, _ = module.step(x, module.reset_state())
    assert first.shape == (2, 8)
    assert not torch.allclose(first, second)
    assert torch.allclose(first, reset_first)


@pytest.mark.parametrize("name", ("lstm", "stc", "qformer", "transformer"))
def test_streaming_reset_reproduces_first_step(name):
    torch.manual_seed(22)
    module = _build(name).eval()
    x = torch.randn(2, 8)
    first, memory = module.step(x)
    assert memory is not None
    reset_first, _ = module.step(x, module.reset_state())
    assert torch.allclose(first, reset_first, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ("stc", "qformer", "transformer"))
def test_explicit_history_modules_are_strictly_causal(name):
    torch.manual_seed(3)
    module = _build(name).eval()
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100.0
    original_states, _ = module.forward_sequence(x)
    changed_states, _ = module.forward_sequence(changed)
    assert torch.allclose(original_states[:, :4], changed_states[:, :4], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ("stc", "qformer", "transformer"))
def test_explicit_history_is_truncated(name):
    module = _build(name, history=3).eval()
    _, memory = module.forward_sequence(torch.randn(2, 7, 8))
    assert memory.shape == (2, 3, 8)
    assert module.get_memory_stats(memory)["num_elements"] == 2 * 3 * 8


def test_ssm_factory_preserves_legacy_state_dict_and_outputs():
    torch.manual_seed(4)
    legacy = _LegacySSM(8, 8, 1, 4).eval()
    rebuilt = _build("ssm").eval()
    rebuilt.load_state_dict(legacy.state_dict())
    x = torch.randn(2, 5, 8)
    old_output, old_memory, old_state = legacy.forward_chunk(x, return_internal=True)
    new_output, new_memory, new_state = rebuilt.forward_chunk(x, return_internal=True)
    adapter = nn.Linear(4, 4).eval()
    score_head = nn.Linear(4, 1).eval()
    old_score = score_head(x[..., :4] + 0.1 * adapter(old_output))
    new_score = score_head(x[..., :4] + 0.1 * adapter(new_output))
    assert list(rebuilt.state_dict()) == list(legacy.state_dict())
    assert torch.equal(old_output, new_output)
    assert torch.equal(old_state, new_state)
    assert torch.equal(old_memory[0], new_memory[0])
    assert torch.equal(old_score, new_score)


def test_baselines_leave_rng_at_ssm_reference_for_shared_module_initialization():
    torch.manual_seed(44)
    _ = build_temporal_module(
        "ssm", d_input=8, d_state=8, output_dim=4, ssm_cls=_LegacySSM,
    )
    expected_shared_weight = nn.Linear(4, 4).weight.detach().clone()

    torch.manual_seed(44)
    _ = build_temporal_module(
        "transformer",
        d_input=8,
        d_state=8,
        output_dim=4,
        history_length=4,
        ssm_cls=_LegacySSM,
        match_ssm_rng=True,
    )
    actual_shared_weight = nn.Linear(4, 4).weight.detach().clone()
    assert torch.equal(actual_shared_weight, expected_shared_weight)


@pytest.mark.parametrize("name", ("lstm", "stc", "qformer", "transformer"))
def test_baseline_state_dict_roundtrip_and_temporal_gradients(name):
    torch.manual_seed(45)
    module = _build(name).train()
    rebuilt = _build(name).train()
    rebuilt.load_state_dict(module.state_dict())
    x = torch.randn(2, 5, 8)
    torch.manual_seed(46)
    expected, _ = module.forward_chunk(x)
    torch.manual_seed(46)
    actual, _ = rebuilt.forward_chunk(x)
    assert torch.allclose(actual, expected)
    actual.square().mean().backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in rebuilt.parameters()
    )


def test_legacy_checkpoint_defaults_to_ssm_and_mismatch_fails():
    warnings = []
    assert checkpoint_temporal_config({}, warn=warnings.append) == ("ssm", 16)
    assert warnings and "legacy ssm" in warnings[0]
    with pytest.raises(ValueError, match="temporal model mismatch"):
        checkpoint_temporal_config(
            {"temporal_model": "lstm", "temporal_history": 16},
            requested_model="transformer",
        )
    with pytest.raises(ValueError, match="temporal history mismatch"):
        checkpoint_temporal_config(
            {"temporal_model": "transformer", "temporal_history": 8},
            requested_model="transformer",
            requested_history=16,
        )
    with pytest.raises(ValueError, match="invalid temporal_history"):
        checkpoint_temporal_config(
            {"temporal_model": "transformer", "temporal_history": 0},
        )


@pytest.mark.parametrize("name", ("lstm", "stc", "qformer", "transformer"))
def test_stage_ap_accepts_shared_readout_hidden_for_each_temporal_module(name):
    module = _build(name).eval()
    shared_readout = nn.Linear(4, 6).eval()
    with torch.no_grad():
        output, _ = module.forward_chunk(torch.randn(1, 7, 8))
        score_hidden = shared_readout(output)
    gt = torch.tensor([0, 0, 1, 1, 1, 0, 0]).numpy()
    rows = [
        {"window_index": index, "start_frame": index, "end_frame": index + 1}
        for index in range(7)
    ]
    records = evaluate_video_stage_ap(
        video_id=name,
        frame_gt=gt,
        window_rows=rows,
        score_hidden=score_hidden[0].numpy(),
    )
    assert records[0]["status"] == "valid"
    assert 0.0 <= records[0]["stage_ap"] <= 1.0
