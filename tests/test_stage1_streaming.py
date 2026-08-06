from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from hivau_dataset import _read_video_label
from stage1_streaming import (
    build_summary_query_batch,
    build_window_infos,
    collect_summary_triggers,
    detach_state_cache,
    score_bce_loss,
    score_metrics_from_logits,
    summary_ce_loss,
)


class _TinyTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.vocab = {"a": 3, "b": 4, "c": 5}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.get(tok, 6) for tok in text.split() if tok]

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id


class _TinyLM(nn.Module):
    def __init__(self, hidden=4, vocab=8):
        super().__init__()
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, inputs_embeds, attention_mask=None, labels=None, **kwargs):
        logits = self.proj(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"logits": logits, "loss": loss, "hidden_states": [inputs_embeds]})


def test_window_soft_labels_and_tail_denominator():
    infos = build_window_infos(
        n_frames=55,
        fps=10.0,
        events=[[0.0, 0.0], [0.0, 2.0], [3.0, 3.9], [3.5, 4.2]],
        frames_per_clip=4,
        sample_interval=5,
    )
    assert infos[0].soft_label == 1.0
    assert infos[1].soft_label == 0.5
    assert infos[2].soft_label == 1 / 3

    no_overlap = build_window_infos(
        n_frames=20,
        fps=10.0,
        events=[],
        frames_per_clip=4,
        sample_interval=1,
    )
    assert no_overlap[0].soft_label == 0.0


def test_summary_trigger_fires_when_clip_end_falls_inside_window():
    infos = build_window_infos(
        n_frames=60,
        fps=10.0,
        events=[],
        frames_per_clip=4,
        sample_interval=5,
        summary_clips=[
            {"clip_id": "aligned", "clip_start_frame": 0, "clip_end_frame": 20, "text": "a b"},
            {"clip_id": "mid", "clip_start_frame": 0, "clip_end_frame": 35, "text": "c"},
            {"clip_id": "also", "clip_start_frame": 10, "clip_end_frame": 40, "text": "a"},
        ],
    )
    assert [t["clip_id"] for t in infos[0].summary_triggers] == ["aligned"]
    assert infos[1].skipped_summary_boundaries == 0
    assert [t["clip_id"] for t in infos[1].summary_triggers] == ["mid", "also"]


def test_summary_trigger_mid_window_example_192_239_clip_end_201():
    infos = build_window_infos(
        n_frames=300,
        fps=30.0,
        events=[],
        frames_per_clip=16,
        sample_interval=3,
        summary_clips=[
            {"clip_id": "c201", "clip_start_frame": 100, "clip_end_frame": 201, "text": "a"},
        ],
    )
    trigger_windows = [
        info for info in infos if any(t["clip_id"] == "c201" for t in info.summary_triggers)
    ]
    assert len(trigger_windows) == 1
    assert trigger_windows[0].start_frame == 192
    assert trigger_windows[0].valid_end_frame == 240


def test_one_chunk_multiple_summary_triggers():
    infos = build_window_infos(
        n_frames=400,
        fps=30.0,
        events=[],
        frames_per_clip=16,
        sample_interval=3,
        summary_clips=[
            {"clip_id": "A", "clip_start_frame": 0, "clip_end_frame": 100, "text": "a"},
            {"clip_id": "B", "clip_start_frame": 0, "clip_end_frame": 245, "text": "b"},
            {"clip_id": "C", "clip_start_frame": 20, "clip_end_frame": 260, "text": "c"},
        ],
    )
    assert [t["clip_id"] for t in infos[2].summary_triggers] == ["A"]
    assert [t["clip_id"] for t in infos[5].summary_triggers] == ["B", "C"]


def test_clip_cross_chunk_triggers_only_in_second_chunk():
    infos = build_window_infos(
        n_frames=700,
        fps=30.0,
        events=[],
        frames_per_clip=16,
        sample_interval=3,
        summary_clips=[
            {"clip_id": "cross", "clip_start_frame": 50, "clip_end_frame": 430, "text": "a"},
        ],
    )
    chunk1 = infos[:8]
    chunk2 = infos[8:16]
    assert sum(len(w.summary_triggers) for w in chunk1) == 0
    assert sum(len(w.summary_triggers) for w in chunk2) == 1


def test_state_cache_detach_and_video_isolation():
    state = {
        "v1": {0: torch.ones(1, requires_grad=True) * 2},
        "v2": {0: torch.ones(1, requires_grad=True) * 3},
    }
    detached = detach_state_cache(state)
    assert detached["v1"][0].grad_fn is None
    assert detached["v2"][0].grad_fn is None
    detached.pop("v1")
    assert "v2" in detached
    assert "v1" not in detached


def test_score_loss_and_metrics_keep_soft_labels():
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[0.2, 0.8, -1.0]])
    valid = torch.tensor([[True, True, False]])
    loss = score_bce_loss(logits, targets, valid)
    loss.backward()
    metrics = score_metrics_from_logits(logits.detach(), targets, valid, binary_threshold=0.5)
    assert loss.isfinite()
    assert metrics["soft_targets"].tolist() == [0.2, 0.8]
    assert metrics["binary_targets"].tolist() == [0, 1]
    assert logits.grad is not None


def test_summary_batch_masks_and_zero_loss_without_triggers():
    tok = _TinyTokenizer()
    embed = nn.Embedding(8, 4)
    states = torch.randn(2, 4)
    query = nn.Parameter(torch.randn(1, 4))
    batch = build_summary_query_batch(embed, tok, states, query, ["a b", "c"])
    assert batch["labels"][:, :2].eq(-100).all()
    assert batch["caption_token_count"] == 5

    lm = _TinyLM()
    zero, info = summary_ce_loss(lm, embed, tok, states[:0], query, [])
    assert zero.item() == 0.0
    assert info["num_summary_triggers"] == 0


def test_summary_query_gradient_and_average_loss():
    tok = _TinyTokenizer()
    embed = nn.Embedding(8, 4)
    lm = _TinyLM()
    states = torch.randn(2, 4, requires_grad=True)
    query = nn.Parameter(torch.randn(1, 4))
    loss, info = summary_ce_loss(lm, embed, tok, states, query, ["a", "b c"])
    loss.backward()
    assert info["num_summary_triggers"] == 2
    assert query.grad is not None
    assert query.grad.abs().sum().item() > 0


def test_score_query_gradient_from_score_loss():
    score_query = nn.Parameter(torch.randn(1, 4))
    score_head = nn.Linear(4, 1)
    state = torch.randn(1, 4)
    logits = score_head(state + score_query).view(1, 1)
    targets = torch.tensor([[1.0]])
    valid = torch.tensor([[True]])
    loss = score_bce_loss(logits, targets, valid)
    loss.backward()
    assert score_query.grad is not None
    assert score_query.grad.abs().sum().item() > 0


def test_queries_optimizer_and_checkpoint_round_trip(tmp_path):
    class TinyStage1(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_query = nn.Parameter(torch.randn(1, 4))
            self.summary_query = nn.Parameter(torch.randn(1, 4))
            self.score_head = nn.Linear(4, 1)

        def score(self, x):
            return self.score_head(x + self.score_query).squeeze(-1)

    model = TinyStage1()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert id(model.score_query) in param_ids
    assert id(model.summary_query) in param_ids

    x = torch.randn(3, 4)
    score_before = model.score(x).detach()
    path = Path(tmp_path) / "state.pt"
    torch.save({
        "score_query": model.score_query.detach().clone(),
        "summary_query": model.summary_query.detach().clone(),
        "score_head": model.score_head.state_dict(),
    }, path)
    loaded = TinyStage1()
    state = torch.load(path, map_location="cpu")
    loaded.score_query.data.copy_(state["score_query"])
    loaded.summary_query.data.copy_(state["summary_query"])
    loaded.score_head.load_state_dict(state["score_head"])
    assert torch.allclose(score_before, loaded.score(x))


def test_collect_summary_triggers_ignores_padding():
    batch = {
        "summary_triggers": [[
            [{"text": "a", "clip_id": "c0"}],
            [{"text": "b", "clip_id": "c1"}],
        ]],
        "skipped_summary_boundaries": [[0, 2]],
    }
    valid = torch.tensor([[True, False]])
    triggers, skipped = collect_summary_triggers(batch, valid)
    assert triggers == [(0, 0, {"text": "a", "clip_id": "c0"})]
    assert skipped == 2


def test_no_summary_trigger_zero_loss():
    tok = _TinyTokenizer()
    embed = nn.Embedding(8, 4)
    lm = _TinyLM()
    states = torch.randn(0, 4)
    query = nn.Parameter(torch.randn(1, 4))
    loss, info = summary_ce_loss(lm, embed, tok, states, query, [])
    assert loss.item() == 0.0
    assert info["num_summary_triggers"] == 0


def test_official_hivau_label_formats():
    empty_frames = np.zeros(4, dtype=np.uint8)
    assert _read_video_label({"label": ["Normal"]}, empty_frames) == 0
    assert _read_video_label({"label": ["Abuse"]}, empty_frames) == 1
    assert _read_video_label({"label": "Normal"}, empty_frames) == 0
    assert _read_video_label({"label": "Abuse"}, empty_frames) == 1
    assert _read_video_label({"label": 0}, empty_frames) == 0
    assert _read_video_label({"label": 1}, empty_frames) == 1
