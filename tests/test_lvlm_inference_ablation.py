import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.benchmark_lvlm_inference import (  # noqa: E402
    aggregate_latency,
    build_generation_inputs,
    handle_video_failure,
    max_new_tokens_for_mode,
    maybe_generate_texts,
    maybe_generate_oracle_transition_texts,
    oracle_transition_type,
    parallel_readout_texts,
    score_single_window,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [" ".join(str(int(x)) for x in row) for row in sequences]


class _Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.generate_calls = []

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.embed

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        n = kwargs["inputs_embeds"].shape[0]
        assert n == 1
        max_new_tokens = kwargs["max_new_tokens"]
        return torch.ones(n, max_new_tokens, dtype=torch.long)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = _Qwen()
        self.score_query = nn.Parameter(torch.randn(1, 4))
        self.summary_query = nn.Parameter(torch.randn(1, 4))
        self.score_head = nn.Linear(4, 1)
        self.forward_batch_sizes = []

    def forward_score_token(self, states, embed_fn, tokenizer, prompt_text):
        self.forward_batch_sizes.append(states.shape[0])
        return states.float().sum(dim=-1)

    def forward_summary_query_hidden(self, states, embed_fn):
        return states + 1.0


def test_forward_mode_does_not_call_generate_and_generation_time_is_zero():
    model = _Model()
    tokenizer = _Tokenizer()
    states = torch.randn(2, 4)
    assert max_new_tokens_for_mode("score-only") is None
    texts = maybe_generate_texts(
        model,
        model.qwen.get_input_embeddings(),
        tokenizer,
        states,
        "Current video status:",
        max_new_tokens_for_mode("score-only"),
    )
    assert texts == []
    assert model.qwen.generate_calls == []
    metrics = aggregate_latency([
        {"latency_ms": 4.0, "generation_time_ms": 0.0, "window_duration_sec": 1.0, "is_warmup": False}
    ])
    assert metrics["generation_time_ms"] == 0.0


def test_ar16_and_ar64_generation_arguments_are_fixed():
    for mode, expected_tokens in [("ar-16", 16), ("ar-64", 64), ("oracle-transition-ar-16", 16)]:
        model = _Model()
        tokenizer = _Tokenizer()
        states = torch.randn(1, 4)
        texts = maybe_generate_texts(
            model,
            model.qwen.get_input_embeddings(),
            tokenizer,
            states,
            "Current video status:",
            max_new_tokens_for_mode(mode),
        )
        assert len(texts) == 1
        call = model.qwen.generate_calls[-1]
        assert call["inputs_embeds"].shape[0] == 1
        assert call["max_new_tokens"] == expected_tokens
        assert call["do_sample"] is False
        assert call["num_beams"] == 1
        assert call["use_cache"] is True


def test_oracle_transition_type_skips_first_and_same_state_windows():
    assert oracle_transition_type(None, 0) is None
    assert oracle_transition_type(None, 1) is None
    assert oracle_transition_type(0, 0) is None
    assert oracle_transition_type(1, 1) is None
    assert oracle_transition_type(0, 1) == "anomaly_onset"
    assert oracle_transition_type(1, 0) == "anomaly_offset"


def test_oracle_transition_generation_only_on_gt_state_changes():
    model = _Model()
    tokenizer = _Tokenizer()
    embed = model.qwen.get_input_embeddings()
    states = torch.randn(1, 4)
    sequence = [0, 0, 1, 1, 0]
    previous = None
    transitions = []
    for current in sequence:
        texts, token_count, transition = maybe_generate_oracle_transition_texts(
            model,
            embed,
            tokenizer,
            states,
            "Current video status:",
            previous,
            current,
        )
        if transition is None:
            assert texts == []
            assert token_count == 0
        else:
            transitions.append(transition)
            assert len(texts) == 1
            assert token_count == 16
        previous = current

    assert transitions == ["anomaly_onset", "anomaly_offset"]
    assert len(model.qwen.generate_calls) == 2
    assert [call["max_new_tokens"] for call in model.qwen.generate_calls] == [16, 16]


def test_anomaly_score_is_identical_across_modes():
    model = _Model()
    tokenizer = _Tokenizer()
    embed = model.qwen.get_input_embeddings()
    states = torch.randn(5, 4)
    scores = {}
    for mode in ["score-only", "ar-16", "ar-64"]:
        mode_scores = []
        for state in states:
            one_state = state.unsqueeze(0)
            logits = score_single_window(model, embed, tokenizer, one_state, "Current video status:")
            _ = maybe_generate_texts(
                model,
                embed,
                tokenizer,
                one_state,
                "Current video status:",
                max_new_tokens_for_mode(mode),
            )
            mode_scores.append(logits.detach())
        scores[mode] = torch.cat(mode_scores)
    assert torch.equal(scores["score-only"], scores["ar-16"])
    assert torch.equal(scores["score-only"], scores["ar-64"])
    assert set(model.forward_batch_sizes) == {1}


def test_warmup_records_are_excluded_from_latency_aggregate():
    records = [
        {"latency_ms": 1000.0, "window_duration_sec": 1.0, "is_warmup": True, "generation_time_ms": 0.0},
        {"latency_ms": 10.0, "window_duration_sec": 1.0, "is_warmup": False, "generation_time_ms": 2.0},
        {"latency_ms": 20.0, "window_duration_sec": 1.0, "is_warmup": False, "generation_time_ms": 4.0},
    ]
    metrics = aggregate_latency(records)
    assert metrics["latency_mean_ms"] == 15.0
    assert metrics["measured_windows"] == 2
    assert metrics["throughput_windows_per_sec"] == 2 / 0.03
    assert metrics["generation_time_ms"] == 3.0


def test_generation_inputs_match_summary_ce_prefix():
    model = _Model()
    tokenizer = _Tokenizer()
    embed = model.qwen.get_input_embeddings()
    model.score_query.data.fill_(float("nan"))
    model.summary_query.data.fill_(1.25)
    states = torch.randn(1, 4)
    gen_inputs = build_generation_inputs(model, embed, tokenizer, states, "Current video status:")
    assert gen_inputs["inputs_embeds"].shape == (1, 2, 4)  # state + summary_query
    assert torch.isfinite(gen_inputs["inputs_embeds"]).all()
    assert torch.equal(gen_inputs["inputs_embeds"][:, 1], model.summary_query.expand(1, -1))


def test_parallel_readout_texts_does_not_call_generate():
    class _Readout(nn.Module):
        def forward(self, z_sem, output_weight=None):
            logits = torch.zeros(z_sem.shape[0], 16, 32)
            logits[:, :, 1] = 1.0
            return logits

    model = _Model()
    tokenizer = _Tokenizer()
    texts = parallel_readout_texts(
        model,
        _Readout(),
        model.qwen.get_input_embeddings(),
        tokenizer,
        torch.randn(1, 4),
    )
    assert texts
    assert model.qwen.generate_calls == []


def test_generation_rejects_batched_windows():
    model = _Model()
    tokenizer = _Tokenizer()
    states = torch.randn(2, 4)
    try:
        maybe_generate_texts(
            model,
            model.qwen.get_input_embeddings(),
            tokenizer,
            states,
            "Current video status:",
            16,
        )
    except ValueError as exc:
        assert "batch size 1" in str(exc)
    else:
        raise AssertionError("batched generation should fail")


def test_failed_video_defaults_to_fail_fast():
    failed = []
    exc = RuntimeError("boom")
    try:
        handle_video_failure("video", exc, failed, skip_failed_videos=False)
    except RuntimeError as raised:
        assert raised is exc
    else:
        raise AssertionError("failed video should raise by default")
    assert failed == []
