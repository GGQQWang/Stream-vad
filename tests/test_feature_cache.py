import json
from pathlib import Path

import pytest
import torch

from feature_cache import (
    build_feature_cache_metadata,
    load_feature_cache,
    save_feature_cache_atomic,
)
from hivau_dataset import HIVAUDataset
from hivau_sampler import VideoPairSampler


def _write_annotation(path: Path) -> None:
    path.write_text(json.dumps({
        "normal_vid": {
            "n_frames": 45,
            "fps": 30.0,
            "video_label": 0,
            "events": [],
        },
        "abnormal_vid": {
            "n_frames": 45,
            "fps": 30.0,
            "video_label": 1,
            "events": [[1.34, 1.5]],
        },
    }))


def _write_cache(root: Path, video_id: str, model_id: str = "fake-qwen") -> torch.Tensor:
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    metadata = build_feature_cache_metadata(
        video_id=video_id,
        n_windows=3,
        n_frames=45,
        fps=30.0,
        frames_per_clip=20,
        sample_interval=1,
        min_pixels=128,
        max_pixels=128,
        model_id=model_id,
    )
    save_feature_cache_atomic(
        root,
        video_id=video_id,
        compressed_features=features.to(torch.float16),
        metadata=metadata,
    )
    return features


def test_feature_cache_dataset_does_not_need_video_reader_or_video_files(tmp_path):
    ann = tmp_path / "ann.json"
    cache_root = tmp_path / "cache"
    _write_annotation(ann)
    expected = _write_cache(cache_root, "normal_vid")
    _write_cache(cache_root, "abnormal_vid")

    ds = HIVAUDataset(
        ann,
        tmp_path / "missing_videos",
        total_sampled_frames=20,
        max_windows=2,
        feature_cache_root=cache_root,
        feature_cache_model_id="fake-qwen",
        min_pixels=128,
        max_pixels=128,
    )

    first = ds[0]
    second = ds[1]
    assert "features" in first
    assert "frames" not in first
    assert torch.allclose(first["features"], expected[:2].float())
    assert torch.allclose(second["features"][0], expected[2].float())
    assert torch.allclose(second["features"][1], torch.zeros(4))
    assert second["valid_mask"].tolist() == [True, False]


def test_feature_cache_preserves_window_order(tmp_path):
    ann = tmp_path / "ann.json"
    cache_root = tmp_path / "cache"
    _write_annotation(ann)
    _write_cache(cache_root, "normal_vid")
    _write_cache(cache_root, "abnormal_vid")
    ds = HIVAUDataset(
        ann,
        tmp_path / "missing_videos",
        total_sampled_frames=20,
        max_windows=2,
        feature_cache_root=cache_root,
        feature_cache_model_id="fake-qwen",
        min_pixels=128,
        max_pixels=128,
    )
    assert ds[0]["chunk_start"] == 0
    assert ds[1]["chunk_start"] == 2
    assert ds[0]["features"][0, 0].item() == 0.0
    assert ds[0]["features"][1, 0].item() == 4.0
    assert ds[1]["features"][0, 0].item() == 8.0


def test_feature_cache_config_mismatch_raises(tmp_path):
    ann = tmp_path / "ann.json"
    cache_root = tmp_path / "cache"
    _write_annotation(ann)
    _write_cache(cache_root, "normal_vid", model_id="old-model")
    _write_cache(cache_root, "abnormal_vid")

    with pytest.raises(ValueError, match="metadata mismatch"):
        HIVAUDataset(
            ann,
            tmp_path / "missing_videos",
            total_sampled_frames=20,
            max_windows=2,
            feature_cache_root=cache_root,
            feature_cache_model_id="fake-qwen",
            min_pixels=128,
            max_pixels=128,
        )


def test_cached_and_online_feature_values_can_match_with_fp16_tolerance(tmp_path):
    cache_root = tmp_path / "cache"
    online = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.01]])
    metadata = build_feature_cache_metadata(
        video_id="v",
        n_windows=2,
        n_frames=40,
        fps=30.0,
        frames_per_clip=20,
        sample_interval=1,
        min_pixels=128,
        max_pixels=128,
        model_id="fake-qwen",
    )
    save_feature_cache_atomic(
        cache_root,
        video_id="v",
        compressed_features=online.to(torch.float16),
        metadata=metadata,
    )
    cached = load_feature_cache(
        cache_root,
        video_id="v",
        n_windows=2,
        n_frames=40,
        fps=30.0,
        frames_per_clip=20,
        sample_interval=1,
        min_pixels=128,
        max_pixels=128,
        model_id="fake-qwen",
    )["compressed_features"].float()
    assert torch.allclose(cached, online, atol=5e-3, rtol=5e-3)


def test_pipeline_cache_chunk_encoder_skips_processor_and_vit():
    from pipeline_stage1 import _encode_chunk_states

    class _FakeModel:
        llm_hidden = 4

        def __init__(self):
            self.seen_states = []

        def encode_window_features(self, window_batch, valid_mask, chunk_video_ids, ssm_state_cache, training=True):
            vid = chunk_video_ids[0]
            self.seen_states.append(ssm_state_cache.get(vid))
            ssm_state_cache[vid] = {"state": len(self.seen_states)}
            return window_batch + 1.0, window_batch, ssm_state_cache

    class _BadProcessor:
        @property
        def image_processor(self):
            raise AssertionError("processor should not be used in cache mode")

    batch1 = {
        "features": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]),
        "binary": torch.tensor([[0.0, 0.0]]),
        "valid_mask": torch.tensor([[True, True]]),
        "video_id": ["v"],
        "chunk_start": [0],
    }
    batch2 = {
        "features": torch.tensor([[[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        "binary": torch.tensor([[1.0, -1.0]]),
        "valid_mask": torch.tensor([[True, False]]),
        "video_id": ["v"],
        "chunk_start": [2],
    }
    model = _FakeModel()
    cache = {}
    states1, _, idx1, _, cache = _encode_chunk_states(
        model, _BadProcessor(), batch1, torch.device("cpu"), torch.float32, cache, training=True,
    )
    states2, _, idx2, _, cache = _encode_chunk_states(
        model, _BadProcessor(), batch2, torch.device("cpu"), torch.float32, cache, training=True,
    )
    assert torch.allclose(states1[:, 0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(states2[:, 0], torch.tensor([4.0]))
    assert idx1.tolist() == [0, 1]
    assert idx2.tolist() == [2]
    assert model.seen_states[0] is None
    assert model.seen_states[1] == {"state": 1}


def test_cached_mil_smoke_backward_and_checkpoint(tmp_path):
    ann = tmp_path / "ann.json"
    cache_root = tmp_path / "cache"
    _write_annotation(ann)
    _write_cache(cache_root, "normal_vid")
    _write_cache(cache_root, "abnormal_vid")
    ds = HIVAUDataset(
        ann,
        tmp_path / "missing_videos",
        total_sampled_frames=20,
        max_windows=2,
        feature_cache_root=cache_root,
        feature_cache_model_id="fake-qwen",
        min_pixels=128,
        max_pixels=128,
    )
    sampler = VideoPairSampler(ds.samples, shuffle=False)
    normal_vid, normal_indices, abnormal_vid, abnormal_indices = next(sampler.iter_epoch(0))
    assert normal_vid == "normal_vid"
    assert abnormal_vid == "abnormal_vid"

    scale = torch.nn.Parameter(torch.tensor(0.1))
    alpha_logit = torch.nn.Parameter(torch.tensor(-2.1972246))
    optimizer = torch.optim.SGD([scale, alpha_logit], lr=0.01)

    state_cache = {}

    def _video_scores(indices):
        scores = []
        for idx in indices:
            sample = ds[idx]
            vid = sample["video_id"]
            prev_state = state_cache.get(vid, torch.zeros(1))
            valid = sample["valid_mask"]
            feats = sample["features"][valid]
            scores.append(feats[:, 0] * scale + prev_state + torch.sigmoid(alpha_logit))
            state_cache[vid] = prev_state + feats[:, 0].sum().detach().reshape(1)
        return torch.cat(scores)

    normal_scores = _video_scores(normal_indices)
    state_cache.pop(normal_vid, None)
    abnormal_scores = _video_scores(abnormal_indices)
    state_cache.pop(abnormal_vid, None)
    loss = normal_scores.square().mean() + torch.relu(0.5 - abnormal_scores.max() + normal_scores.max())
    loss.backward()
    optimizer.step()

    ckpt = tmp_path / "train_state.pt"
    torch.save({
        "adapter": {"scale": scale.detach().clone()},
        "ssm": {},
        "alpha_logit": alpha_logit.detach().clone(),
        "objective": "mil_rank",
    }, ckpt)
    loaded = torch.load(ckpt, map_location="cpu")
    assert torch.isfinite(loss)
    assert scale.grad is not None
    assert alpha_logit.grad is not None
    assert loaded["objective"] == "mil_rank"
    assert torch.allclose(loaded["alpha_logit"], alpha_logit.detach())
