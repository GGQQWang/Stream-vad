"""CPU-only unit tests for Stage-1 generation pipeline.

Run:  python -m pytest tests/test_stage1_generation.py -v
   or: python tests/test_stage1_generation.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# minimal mock tokenizer
# ---------------------------------------------------------------------------

class _MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.vocab_size = 256
        self.unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == "Normal":
            return [100]
        elif text == "Abnormal":
            return [200, 201]  # two tokens
        elif text == "Current video status:":
            return [10, 11, 12]
        return [0]

    def decode(self, ids):
        id_to_text = {100: "Normal", 200: "Ab", 201: "normal", 10: "Cur", 11: "rent", 12: ":"}
        return "".join(id_to_text.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_build_generation_batch_basic():
    """Normal/Abnormal targets mapped to correct answer sequences."""
    from pipeline_stage1 import build_status_generation_batch

    tokenizer = _MockTokenizer()
    embed_fn = nn.Embedding(tokenizer.vocab_size, 64)
    state = torch.randn(3, 64)       # 3 samples
    targets = torch.tensor([0, 1, 0])

    batch = build_status_generation_batch(embed_fn, tokenizer, state, targets)

    # Normal sample (target=0): answer_ids = [100, 2] (Normal + eos)
    # Abnormal sample (target=1): answer_ids = [200, 201, 2]
    lbl0 = batch["labels"][0]
    lbl1 = batch["labels"][1]

    # answer positions should not be -100
    answer_positions_0 = (lbl0 != -100).nonzero(as_tuple=True)[0]
    answer_positions_1 = (lbl1 != -100).nonzero(as_tuple=True)[0]
    assert len(answer_positions_0) == 2  # Normal token + eos
    assert len(answer_positions_1) == 3  # two Abnormal tokens + eos
    assert lbl0[answer_positions_0[0]].item() == 100   # Normal token
    assert lbl0[answer_positions_0[1]].item() == 2     # eos
    assert lbl1[answer_positions_1[0]].item() == 200
    assert lbl1[answer_positions_1[1]].item() == 201

    # state + prompt positions should be -100
    state_and_prompt = lbl0[:1 + 3]  # state token + 3 prompt tokens
    assert (state_and_prompt == -100).all()

    # padding labels should be -100
    max_len = batch["inputs_embeds"].shape[1]
    for i in range(3):
        actual_len = batch["attention_mask"][i].sum().item()
        assert (batch["labels"][i, actual_len:] == -100).all()


def test_padding_handles_different_lengths():
    """Two samples with different answer lengths pad correctly."""
    from pipeline_stage1 import build_status_generation_batch

    tokenizer = _MockTokenizer()
    embed_fn = nn.Embedding(tokenizer.vocab_size, 64)
    state = torch.randn(2, 64)
    targets = torch.tensor([0, 1])

    batch = build_status_generation_batch(embed_fn, tokenizer, state, targets)
    assert batch["inputs_embeds"].shape[0] == 2
    # max_len should be the Abnormal sample (longer)
    assert batch["inputs_embeds"].shape[1] == 1 + 3 + 3  # state + 3 prompt + (2 Abnormal tokens + eos)


def test_masked_token_ce():
    """Per-sample CE normalises correctly."""
    from pipeline_stage1 import masked_token_ce

    N, L, V = 2, 5, 10
    logits = torch.randn(N, L, V)
    labels = torch.full((N, L), -100, dtype=torch.long)
    labels[0, 3] = 3
    labels[0, 4] = 4          # sample 0: 2 answer tokens
    labels[1, 3] = 5          # sample 1: 1 answer token
    answer_mask = labels != -100
    targets = torch.tensor([0, 1])

    loss, info = masked_token_ce(logits, labels, answer_mask, targets, abnormal_loss_weight=2.0)

    assert not torch.isnan(loss)
    assert info["n_samples"] == 2
    assert info["n_answer_tokens"] == 3


def test_anomaly_score_direction():
    """Higher NLL(normal) - NLL(abnormal) → more likely abnormal."""
    normal_nll = torch.tensor([0.5, 2.0, 0.3])
    abnormal_nll = torch.tensor([2.0, 0.1, 0.8])
    score = normal_nll - abnormal_nll
    # sample 0: 0.5 - 2.0 = -1.5 < 0 → normal
    # sample 1: 2.0 - 0.1 = 1.9 > 0 → abnormal
    # sample 2: 0.3 - 0.8 = -0.5 < 0 → normal
    pred = (score > 0).int()
    assert pred[0].item() == 0
    assert pred[1].item() == 1
    assert pred[2].item() == 0


def test_state_prompt_labels_masked():
    """state and prompt positions get -100 labels."""
    from pipeline_stage1 import build_status_generation_batch

    tokenizer = _MockTokenizer()
    embed_fn = nn.Embedding(tokenizer.vocab_size, 64)
    state = torch.randn(1, 64)
    targets = torch.tensor([0])

    batch = build_status_generation_batch(embed_fn, tokenizer, state, targets)
    labels = batch["labels"][0]

    # First position = state token → -100
    assert labels[0].item() == -100
    # Next 3 positions = prompt → -100
    assert (labels[1:4] == -100).all()
    # After that = answer + eos → NOT -100
    assert (labels[4:4+2] != -100).all()


def test_stage1_score_components_defined_on_instances():
    """Stage 1 score-token objective owns explicit trainable heads/queries."""
    from pipeline_stage1 import StreamingVADGenerationModel
    assert "score_head" not in StreamingVADGenerationModel.__dict__
    assert "future_head" not in StreamingVADGenerationModel.__dict__
    assert "score_query" not in StreamingVADGenerationModel.__dict__
    assert "summary_query" not in StreamingVADGenerationModel.__dict__


def test_stage2_not_broken():
    """Stage 2 file should still compile."""
    import py_compile
    stage2_path = PROJECT_ROOT / "pipeline_stage2_detection.py"
    py_compile.compile(str(stage2_path), doraise=True)


if __name__ == "__main__":
    test_build_generation_batch_basic()
    test_padding_handles_different_lengths()
    test_masked_token_ce()
    test_anomaly_score_direction()
    test_state_prompt_labels_masked()
    test_stage1_score_components_defined_on_instances()
    test_stage2_not_broken()
    print("ALL CPU TESTS PASSED")
