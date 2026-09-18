import json

import pytest
import torch
import torch.nn as nn

from long_horizon_scaling import (
    HISTORY_HORIZONS,
    collect_scaling_results,
    validate_long64_checkpoint,
)
from temporal_modules import (
    StrictHorizonTemporalRunner,
    build_temporal_module,
    truncate_explicit_temporal_memory,
)


class _ToyRecurrent(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.temporal_model = name
        self.history_length = None if name == "ssm" else 64
        self.output_dim = 3
        self.state_dim = 3
        self.weight = nn.Parameter(torch.ones(()))

    def forward_chunk(self, inputs, state=None, return_internal=False):
        initial = inputs.new_zeros(inputs.shape[0], inputs.shape[2]) if state is None else state
        internal = torch.cumsum(inputs * self.weight, dim=1) + initial.unsqueeze(1)
        new_state = internal[:, -1]
        if return_internal:
            return internal, new_state, internal
        return internal, new_state


def _build_explicit(name, history=64):
    return build_temporal_module(
        name,
        d_input=8,
        d_state=8,
        output_dim=4,
        history_length=history,
    ).eval()


@pytest.mark.parametrize("name", ("qformer", "transformer"))
def test_learned_position_temporal_models_support_history_64(name):
    module = _build_explicit(name)
    assert module.history_length == 64
    assert module.position.shape == (64, 8)
    with torch.no_grad():
        _, memory, internal = module.forward_chunk(
            torch.randn(1, 64, 8), return_internal=True,
        )
    assert memory.shape == (1, 64, 8)
    assert internal.shape == (1, 64, 8)


@pytest.mark.parametrize("horizon", HISTORY_HORIZONS)
def test_explicit_history_is_truncated_to_requested_horizon(horizon):
    module = _build_explicit("stc")
    runner = StrictHorizonTemporalRunner(module, horizon, max_horizon=64)
    with torch.no_grad():
        output, internal = runner.forward_chunk(torch.randn(1, 80, 8))
    assert output.shape == (1, 80, 4)
    assert internal.shape == (1, 80, 8)
    assert runner.retained_input_count == horizon


def test_horizon_above_checkpoint_capacity_is_rejected():
    module = _build_explicit("transformer")
    with pytest.raises(ValueError, match="exceeds checkpoint capacity"):
        StrictHorizonTemporalRunner(module, 65, max_horizon=64)


@pytest.mark.parametrize("name", ("qformer", "transformer"))
def test_explicit_native_buffer_size_grows_with_horizon(name):
    module = _build_explicit(name)
    memory = torch.zeros(1, 64, 8)
    sizes = []
    for horizon in HISTORY_HORIZONS:
        retained = truncate_explicit_temporal_memory(module, memory, horizon)
        sizes.append(retained.numel() * retained.element_size())
    assert all(right > left for left, right in zip(sizes, sizes[1:]))


@pytest.mark.parametrize("name", ("qformer", "transformer"))
def test_explicit_strict_horizon_has_no_future_leakage(name):
    torch.manual_seed(11)
    module = _build_explicit(name)
    inputs = torch.randn(1, 12, 8)
    changed = inputs.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 50
    with torch.no_grad():
        first, _ = StrictHorizonTemporalRunner(module, 8, max_horizon=64).forward_chunk(inputs)
        second, _ = StrictHorizonTemporalRunner(module, 8, max_horizon=64).forward_chunk(changed)
    assert torch.allclose(first[:, :7], second[:, :7], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ("ssm", "lstm"))
def test_recurrent_strict_horizon_matches_zero_state_sliding_replay(name):
    module = _ToyRecurrent(name).eval()
    inputs = torch.arange(1, 25, dtype=torch.float32).reshape(1, 8, 3)
    runner = StrictHorizonTemporalRunner(module, 4, max_horizon=64)
    with torch.no_grad():
        output, internal = runner.forward_chunk(inputs)
    expected = torch.stack([
        inputs[:, max(0, index - 3):index + 1].sum(dim=1)
        for index in range(inputs.shape[1])
    ], dim=1)
    assert torch.equal(output, expected)
    assert torch.equal(internal, expected)
    assert runner.num_replayed_inputs == 1 + 2 + 3 + 4 * 5


@pytest.mark.parametrize("name", ("ssm", "lstm"))
def test_recurrent_native_memory_is_not_truncated_by_effectiveness_horizon(name):
    module = _ToyRecurrent(name)
    memory = (torch.zeros(1, 3), torch.zeros(1, 3)) if name == "lstm" else {0: torch.zeros(1, 3)}
    for horizon in HISTORY_HORIZONS:
        assert truncate_explicit_temporal_memory(module, memory, horizon) is memory


def test_runner_reset_prevents_cross_video_state_leakage():
    module = _ToyRecurrent("ssm").eval()
    runner = StrictHorizonTemporalRunner(module, 8, max_horizon=64)
    first = torch.randn(1, 3)
    with torch.no_grad():
        expected, _ = runner.step(first)
        runner.step(torch.randn(1, 3))
        runner.reset()
        actual, _ = runner.step(first)
    assert torch.equal(actual, expected)
    assert runner.num_predictions == 1


def test_long64_recurrent_training_forward_unrolls_beyond_16():
    module = _ToyRecurrent("lstm")
    _, _, internal = module.forward_chunk(torch.randn(1, 64, 3), return_internal=True)
    assert internal.shape[1] == 64


def test_strict_horizon_evaluation_is_deterministic():
    module = _ToyRecurrent("ssm").eval()
    inputs = torch.randn(1, 20, 3)
    with torch.no_grad():
        first, _ = StrictHorizonTemporalRunner(module, 16, max_horizon=64).forward_chunk(inputs)
        second, _ = StrictHorizonTemporalRunner(module, 16, max_horizon=64).forward_chunk(inputs)
    assert torch.equal(first, second)


def test_stc_architecture_keeps_five_window_receptive_field():
    module = _build_explicit("stc")
    assert module.num_layers == 2
    assert module.conv1.kernel_size == (3,)
    assert module.conv2.kernel_size == (3,)
    assert 1 + 2 * (module.conv1.kernel_size[0] - 1) == 5


@pytest.mark.parametrize("name", ("stc", "qformer", "transformer"))
def test_t64_matches_explicit_checkpoint_full_capacity(name):
    torch.manual_seed(19)
    module = _build_explicit(name)
    inputs = torch.randn(1, 64, 8)
    with torch.no_grad():
        expected, _, _ = module.forward_chunk(inputs, return_internal=True)
        actual, _ = StrictHorizonTemporalRunner(
            module, 64, max_horizon=64,
        ).forward_chunk(inputs)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ("ssm", "lstm"))
def test_t64_matches_recurrent_full_64_window_rollout(name):
    module = _ToyRecurrent(name).eval()
    inputs = torch.randn(1, 64, 3)
    with torch.no_grad():
        expected, _, _ = module.forward_chunk(inputs, return_internal=True)
        actual, _ = StrictHorizonTemporalRunner(
            module, 64, max_horizon=64,
        ).forward_chunk(inputs)
    assert torch.equal(actual, expected)


def _long64_state(method):
    return {
        "temporal_model": method,
        "max_training_horizon": 64,
        "max_windows": 64,
        "temporal_history": 64,
        "training_history_reset": "chunk",
        "temporal_state_dim": 256,
        "temporal_config": {
            "name": method,
            "history_length": 64 if method in {"stc", "qformer", "transformer"} else None,
        },
    }


@pytest.mark.parametrize("method", ("ssm", "lstm", "stc", "qformer", "transformer"))
def test_long64_checkpoint_metadata_validation(method):
    validate_long64_checkpoint(_long64_state(method), method)


def test_old_short_horizon_checkpoint_is_rejected():
    state = _long64_state("transformer")
    state["max_windows"] = 8
    with pytest.raises(ValueError, match="max_windows"):
        validate_long64_checkpoint(state, "transformer")


def test_result_collection_enforces_native_buffer_scaling(tmp_path):
    for method in ("ssm", "lstm", "stc", "qformer", "transformer"):
        for index, horizon in enumerate(HISTORY_HORIZONS):
            result_dir = tmp_path / method / f"T{horizon}"
            result_dir.mkdir(parents=True)
            recurrent = method in {"ssm", "lstm"}
            metrics = {
                "temporal_model": method,
                "history_horizon": horizon,
                "max_training_horizon": 64,
                "num_failed_videos": 0,
                "visual_fusion": "state_spatial_film",
                "inference_dtype": "torch.bfloat16",
                "test_manifest": "test.json",
                "feature_cache_root": "cache",
                "profile_warmup_windows": 16,
                "profiling_batch_size": 1,
                "profiling_scope": "cached_visual_feature_to_score_logit",
                "effectiveness_protocol": "strict_sliding_recent_T_temporal_only_replay",
                "representation_k_eval": 16,
                "auc": 0.5,
                "ap": 0.25,
                "mean_latency_ms": 1.0,
                "p95_latency_ms": 1.2,
                "throughput_windows_s": 1000.0,
                "history_buffer_bytes": 128 if recurrent else 128 * (index + 1),
                "history_buffer_mb": 0.001,
                "peak_gpu_memory_gb": 1.0,
                "v_normal_k16": None if horizon < 16 else 0.1,
                "v_anomaly_k16": None if horizon < 16 else 0.2,
                "v_boundary_k16": None if horizon < 16 else 0.3,
            }
            (result_dir / "metrics.json").write_text(json.dumps(metrics))
    rows = collect_scaling_results(tmp_path)
    assert len(rows) == 25
