import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _import_infer(monkeypatch):
    tb_mod = types.ModuleType("torch.utils.tensorboard")
    tb_mod.SummaryWriter = object
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb_mod)

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda model, *args, **kwargs: model
    peft_mod.PeftModel = type(
        "PeftModel",
        (),
        {"from_pretrained": staticmethod(lambda model, *args, **kwargs: model)},
    )
    tr_mod = types.ModuleType("transformers")
    tr_mod.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: object())})
    tr_mod.Qwen2VLProcessor = type("Qwen2VLProcessor", (), {"from_pretrained": staticmethod(lambda *a, **k: object())})
    tr_mod.get_linear_schedule_with_warmup = lambda *args, **kwargs: None
    tr_mod.set_seed = lambda *args, **kwargs: None

    class _Qwen(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 4})()

        def to(self, *args, **kwargs):
            return self

    tr_mod.Qwen2VLForConditionalGeneration = _Qwen
    monkeypatch.setitem(sys.modules, "peft", peft_mod)
    monkeypatch.setitem(sys.modules, "transformers", tr_mod)
    import infer_stage1_ucf as infer
    return importlib.reload(infer)


class _Loadable:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state


class _FakeStage1Model:
    def __init__(self, qwen, d_ssm, llm_hidden, vit_micro_batch, world_include_decoder, visual_fusion):
        self.qwen = qwen
        self.visual_fusion = visual_fusion
        self.ssm = _Loadable()
        self.adapter = _Loadable()
        self.spatial_film = _Loadable()
        self.score_head = _Loadable()
        self.score_query = torch.nn.Parameter(torch.zeros(1, llm_hidden))
        self.summary_query = torch.nn.Parameter(torch.zeros(1, llm_hidden))
        self.alpha_logit = torch.nn.Parameter(torch.zeros(()))
        self.debug_state = True
        self.eval_called = False

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        self.eval_called = True
        return self


def _state(visual_fusion="state_only", include_spatial_film=True):
    out = {
        "frames_per_clip": 16,
        "sample_interval": 3,
        "max_windows": 8,
        "min_pixels": 100352,
        "max_pixels": 100352,
        "objective": "score_token",
        "d_ssm": 4,
        "ssm": {"ssm": torch.ones(1)},
        "adapter": {"adapter": torch.ones(1)},
        "score_head": {"score": torch.ones(1)},
        "score_query": torch.randn(1, 4),
        "summary_query": torch.randn(1, 4),
        "alpha_logit": torch.tensor(0.25),
        "visual_fusion": visual_fusion,
    }
    if include_spatial_film:
        out["spatial_film"] = {"film": torch.ones(1)}
    return out


def _args(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    (stage1_dir / "train_state.pt").write_text("stub")
    (stage1_dir / "lora_adapter").mkdir()
    return type("Args", (), {
        "stage1_dir": str(stage1_dir),
        "model_path": "qwen",
        "device": torch.device("cpu"),
    })()


def test_load_stage1_model_restores_film_spatial_checkpoint(monkeypatch, tmp_path):
    infer = _import_infer(monkeypatch)
    state = _state("film_spatial")
    monkeypatch.setattr(infer.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(infer, "StreamingVADGenerationModel", _FakeStage1Model)
    monkeypatch.setattr(infer, "_load_temporal_conditioning", lambda *args, **kwargs: None)

    model, *_ = infer.load_stage1_model(_args(tmp_path))

    assert model.visual_fusion == "film_spatial"
    assert model.spatial_film.loaded is state["spatial_film"]
    assert model.eval_called


def test_load_stage1_model_film_spatial_requires_spatial_film(monkeypatch, tmp_path):
    infer = _import_infer(monkeypatch)
    monkeypatch.setattr(infer.torch, "load", lambda *args, **kwargs: _state("film_spatial", include_spatial_film=False))
    monkeypatch.setattr(infer, "StreamingVADGenerationModel", _FakeStage1Model)

    with pytest.raises(ValueError, match="missing spatial_film"):
        infer.load_stage1_model(_args(tmp_path))


def test_load_stage1_model_state_only_keeps_old_checkpoint_compatible(monkeypatch, tmp_path):
    infer = _import_infer(monkeypatch)
    state = _state("state_only", include_spatial_film=False)
    state.pop("visual_fusion")
    monkeypatch.setattr(infer.torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(infer, "StreamingVADGenerationModel", _FakeStage1Model)
    monkeypatch.setattr(infer, "_load_temporal_conditioning", lambda *args, **kwargs: None)

    model, *_ = infer.load_stage1_model(_args(tmp_path))

    assert model.visual_fusion == "state_only"
    assert model.spatial_film.loaded is None


class _Ref:
    def __init__(self):
        self.video_id = "v0"
        self.index = 0


def _cached_batch(include_spatial=True):
    batch = {
        "video_id": ["v0"],
        "chunk_start": [0],
        "features": torch.randn(1, 2, 4),
        "valid_mask": torch.tensor([[True, True]]),
        "window_start_frames": torch.tensor([[0, 5]]),
        "valid_end_frames": torch.tensor([[5, 10]]),
    }
    if include_spatial:
        batch["spatial_features"] = torch.randn(1, 2, 3, 4)
        batch["spatial_mask"] = torch.tensor([[[True, False, False], [True, True, False]]])
    return batch


class _Dataset:
    samples = [{
        "video_id": "v0",
        "n_frames": 10,
        "fps": 5.0,
    }]

    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, idx):
        return self.batch


class _InferModel:
    visual_fusion = "film_spatial"

    def __init__(self):
        self.qwen = object()
        self.used_visual_prefix = False
        self.used_state_score = False

    def encode_window_features(self, window_batch, valid_mask, video_ids, ssm_cache, training, return_internal=False):
        assert return_internal
        state = torch.randn(1, 2, 4)
        h_internal = torch.randn(1, 2, 4)
        return state, window_batch, torch.zeros_like(window_batch), h_internal, ssm_cache

    def select_visual_prefix(self, state_emb, valid_b, valid_w, spatial_features, spatial_mask, h_internal):
        assert spatial_features is not None
        assert spatial_mask is not None
        assert h_internal is not None
        self.used_visual_prefix = True
        return torch.randn(2, 2, 4), torch.tensor([[True, False], [True, True]])

    def forward_score_visual_prefix(self, visual_prefix, visual_mask, embed_fn, tokenizer, prompt_text):
        assert self.used_visual_prefix
        return torch.tensor([0.1, -0.2])

    def forward_score_token(self, *args, **kwargs):
        self.used_state_score = True
        raise AssertionError("film_spatial inference must not call forward_score_token")


class _StateOnlyInferModel(_InferModel):
    visual_fusion = "state_only"

    def encode_window_features(self, window_batch, valid_mask, video_ids, ssm_cache, training, return_internal=False):
        assert not return_internal
        state = torch.randn(1, 2, 4)
        return state, window_batch, torch.zeros_like(window_batch), ssm_cache

    def select_visual_prefix(self, *args, **kwargs):
        raise AssertionError("state_only inference must not call select_visual_prefix")

    def forward_score_visual_prefix(self, *args, **kwargs):
        raise AssertionError("state_only inference must not call forward_score_visual_prefix")

    def forward_score_token(self, states, embed_fn, tokenizer, prompt_text):
        self.used_state_score = True
        return torch.tensor([0.1, -0.2])


def test_film_spatial_cached_inference_uses_visual_prefix_path(monkeypatch, tmp_path):
    infer = _import_infer(monkeypatch)
    model = _InferModel()
    monkeypatch.setattr(infer, "hivau_collate", lambda items: items[0])
    monkeypatch.setattr(infer, "load_gt", lambda *args, **kwargs: np.zeros(10, dtype=np.int64))
    monkeypatch.setattr(infer, "save_window_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer, "_find_embed", lambda qwen: torch.nn.Embedding(8, 4))

    result = infer.infer_video(
        model=model,
        processor=None,
        tokenizer=type("Tok", (), {})(),
        dataset=_Dataset(_cached_batch(include_spatial=True)),
        refs=[_Ref()],
        device=torch.device("cpu"),
        dtype=torch.float32,
        prompt_text="prompt",
        output_dir=tmp_path,
        gt_root=tmp_path,
        debug_state=False,
    )

    assert result["num_windows"] == 2
    assert model.used_visual_prefix
    assert not model.used_state_score


def test_state_only_cached_inference_uses_original_score_token_path(monkeypatch, tmp_path):
    infer = _import_infer(monkeypatch)
    model = _StateOnlyInferModel()
    monkeypatch.setattr(infer, "hivau_collate", lambda items: items[0])
    monkeypatch.setattr(infer, "load_gt", lambda *args, **kwargs: np.zeros(10, dtype=np.int64))
    monkeypatch.setattr(infer, "save_window_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer, "_find_embed", lambda qwen: torch.nn.Embedding(8, 4))

    result = infer.infer_video(
        model=model,
        processor=None,
        tokenizer=type("Tok", (), {})(),
        dataset=_Dataset(_cached_batch(include_spatial=False)),
        refs=[_Ref()],
        device=torch.device("cpu"),
        dtype=torch.float32,
        prompt_text="prompt",
        output_dir=tmp_path,
        gt_root=tmp_path,
        debug_state=False,
    )

    assert result["num_windows"] == 2
    assert model.used_state_score
    assert not model.used_visual_prefix


def test_film_spatial_cached_inference_missing_spatial_cache_fails(monkeypatch, tmp_path):
    infer = _import_infer(monkeypatch)
    monkeypatch.setattr(infer, "hivau_collate", lambda items: items[0])
    monkeypatch.setattr(infer, "load_gt", lambda *args, **kwargs: np.zeros(10, dtype=np.int64))
    monkeypatch.setattr(infer, "_find_embed", lambda qwen: torch.nn.Embedding(8, 4))

    with pytest.raises(ValueError, match="requires spatial_features/spatial_mask"):
        infer.infer_video(
            model=_InferModel(),
            processor=None,
            tokenizer=type("Tok", (), {})(),
            dataset=_Dataset(_cached_batch(include_spatial=False)),
            refs=[_Ref()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            prompt_text="prompt",
            output_dir=tmp_path,
            gt_root=tmp_path,
            debug_state=False,
        )
