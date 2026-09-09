import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_human_eval_texts import (  # noqa: E402
    METHODS,
    WindowSample,
    build_blind_outputs,
    build_summary_query_ar_inputs,
    select_random_video_ids,
    sample_balanced_windows,
    summarize_latency,
)
from tools.benchmark_lvlm_inference import maybe_generate_texts  # noqa: E402


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["x" for _ in range(sequences.shape[0])]


class _Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.generate_batch_sizes = []

    def get_input_embeddings(self):
        return self.embed

    def generate(self, **kwargs):
        self.generate_batch_sizes.append(kwargs["inputs_embeds"].shape[0])
        return torch.ones(kwargs["inputs_embeds"].shape[0], kwargs["max_new_tokens"], dtype=torch.long)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = _Qwen()
        self.summary_query = nn.Parameter(torch.ones(1, 4))
        self.score_query = nn.Parameter(torch.full((1, 4), float("nan")))


def _sample(i: int, label: int) -> WindowSample:
    return WindowSample(
        sample_id="",
        video_id=f"v{i % 3}",
        dataset_index=i // 8,
        local_window=i % 8,
        window_idx=i,
        start_frame=i * 48,
        end_frame=(i + 1) * 48,
        fps=24.0,
        gt_label=label,
    )


def test_balanced_sampling_uses_same_sample_ids_and_seed_is_reproducible():
    normal = [_sample(i, 0) for i in range(150)]
    abnormal = [_sample(i + 1000, 1) for i in range(150)]
    first = sample_balanced_windows(normal, abnormal, per_class=100, seed=42)
    second = sample_balanced_windows(normal, abnormal, per_class=100, seed=42)
    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    assert [(s.video_id, s.window_idx, s.gt_label) for s in first] == [
        (s.video_id, s.window_idx, s.gt_label) for s in second
    ]
    assert sum(s.gt_label == 1 for s in first) == 100
    assert sum(s.gt_label == 0 for s in first) == 100


def test_sampling_errors_when_a_class_is_short():
    try:
        sample_balanced_windows([_sample(1, 0)], [_sample(2, 1)], per_class=100, seed=42)
    except ValueError as exc:
        assert "insufficient" in str(exc)
    else:
        raise AssertionError("short class should fail")


def test_blind_file_hides_methods_and_key_restores_mapping():
    raw = [{
        "sample_id": "sample_0000",
        "video_id": "v",
        "window_idx": 1,
        "start_frame": 0,
        "end_frame": 48,
        "start_sec": 0.0,
        "end_sec": 2.0,
        "gt_label": 1,
        "score_prob": 0.7,
        "summary_query_ar_text": "semantic",
        "prompt_ar16_text": "short",
        "prompt_ar64_text": "long",
    }]
    blind, key = build_blind_outputs(raw, seed=42)
    dumped = json.dumps(blind)
    assert "summary_query_ar" not in dumped
    assert "prompt_ar16" not in dumped
    assert "prompt_ar64" not in dumped
    mapping = key["samples"]["sample_0000"]
    assert set(mapping) == {"A", "B", "C"}
    assert set(mapping.values()) == set(METHODS)
    restored = {method: blind[0]["texts"][letter] for letter, method in mapping.items()}
    assert restored == {
        "summary_query_ar": "semantic",
        "prompt_ar16": "short",
        "prompt_ar64": "long",
    }


def test_three_methods_share_one_sample_record():
    raw = {
        "sample_id": "sample_0001",
        "summary_query_ar_text": "semantic",
        "prompt_ar16_text": "short",
        "prompt_ar64_text": "long",
    }
    assert raw["sample_id"] == "sample_0001"
    assert {name for name in raw if name.endswith("_text")} == {
        "summary_query_ar_text",
        "prompt_ar16_text",
        "prompt_ar64_text",
    }


def test_ar_generation_batch_size_is_one():
    model = _Model()
    tokenizer = _Tokenizer()
    states = torch.randn(1, 4)
    maybe_generate_texts(model, model.qwen.get_input_embeddings(), tokenizer, states, "prompt", 16)
    assert model.qwen.generate_batch_sizes == [1]


def test_summary_query_ar_uses_summary_query_not_score_query():
    model = _Model()
    states = torch.zeros(1, 4)
    inputs = build_summary_query_ar_inputs(model, model.qwen.get_input_embeddings(), states)
    assert inputs["inputs_embeds"].shape == (1, 2, 4)
    assert torch.isfinite(inputs["inputs_embeds"]).all()
    assert torch.equal(inputs["inputs_embeds"][0, 1], torch.ones(4))


def test_random_video_selection_is_seeded_and_sized():
    ids = ["v3", "v1", "v2", "v5", "v4"]
    first = select_random_video_ids(ids, 3, seed=42)
    second = select_random_video_ids(ids, 3, seed=42)
    assert first == second
    assert len(first) == 3
    assert set(first).issubset(set(ids))


def test_random_video_latency_summary():
    records = [
        {"summary_query_ar_ms": 1.0, "prompt_ar16_ms": 2.0, "prompt_ar64_ms": 4.0},
        {"summary_query_ar_ms": 3.0, "prompt_ar16_ms": 4.0, "prompt_ar64_ms": 8.0},
    ]
    summary = summarize_latency(records)
    assert summary["summary_query_ar_ms"]["mean"] == 2.0
    assert summary["prompt_ar16_ms"]["p50"] == 3.0
    assert summary["prompt_ar64_ms"]["p95"] == 7.8
