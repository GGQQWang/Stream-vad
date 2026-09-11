import torch
import torch.nn as nn

from parallel_semantic_readout import (
    ParallelSemanticReadout,
    decode_parallel_tokens,
    encode_parallel_semantic_targets,
    freeze_all_except_parallel_readout,
    parallel_semantic_loss,
)
from stage1_streaming import IGNORE_INDEX


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        table = {"a": 3, "b": 4}
        return [table[t] for t in text.split()]

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{int(x)}" for x in ids]

    def batch_decode(self, sequences, skip_special_tokens=True):
        out = []
        for row in sequences:
            toks = []
            for x in row.tolist():
                if skip_special_tokens and int(x) == self.eos_token_id:
                    break
                toks.append(str(int(x)))
            out.append(" ".join(toks))
        return out


class _BatchDecodeDoesNotTruncateTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def batch_decode(self, sequences, skip_special_tokens=True):
        out = []
        for row in sequences:
            toks = [
                str(int(x))
                for x in row
                if not skip_special_tokens or int(x) not in {self.eos_token_id, self.pad_token_id}
            ]
            out.append(" ".join(toks))
        return out


def test_parallel_semantic_readout_shape():
    readout = ParallelSemanticReadout(8, 32, num_queries=16, decoder_dim=8, num_heads=2)
    logits = readout(torch.randn(3, 8))
    assert logits.shape == (3, 16, 32)
    assert logits.dtype == torch.float32


def test_bfloat16_semantic_state_enters_float32_readout_boundary():
    stage1 = nn.Linear(8, 8).to(dtype=torch.bfloat16)
    readout = ParallelSemanticReadout(8, 32, num_queries=16, decoder_dim=8, num_heads=2)
    z_sem = stage1(torch.randn(3, 8, dtype=torch.bfloat16))
    logits = readout(z_sem)
    targets = torch.randint(0, 32, (3, 16), dtype=torch.long)
    loss, _ = parallel_semantic_loss(logits, targets, eos_token_id=2)
    loss.backward()
    assert logits.shape == (3, 16, 32)
    assert logits.dtype == torch.float32
    assert all(p.grad is None for p in stage1.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in readout.parameters())


def test_parallel_queries_have_distinct_position_roles():
    readout = ParallelSemanticReadout(8, 32, num_queries=16, decoder_dim=8, num_heads=2)
    assert readout.query_embed.shape == (16, 8)
    assert torch.unique(readout.query_embed.detach(), dim=0).shape[0] == 16


def test_parallel_readout_uses_noncausal_encoder_over_semantic_and_queries():
    readout = ParallelSemanticReadout(8, 32, num_queries=16, decoder_dim=8, num_heads=2)
    seen = {}
    original_forward = readout.encoder.forward

    def wrapped(src, mask=None, src_key_padding_mask=None, is_causal=None):
        seen["shape"] = tuple(src.shape)
        seen["mask"] = mask
        seen["is_causal"] = is_causal
        return original_forward(
            src,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )

    readout.encoder.forward = wrapped
    logits = readout(torch.randn(2, 8))
    assert seen["shape"] == (2, 17, 8)
    assert seen["mask"] is None
    assert seen["is_causal"] in (None, False)
    assert logits.shape == (2, 16, 32)


def test_pad_positions_do_not_contribute_to_loss():
    logits = torch.zeros(1, 4, 8, requires_grad=True)
    logits.data[0, 0, 3] = 2.0
    logits.data[0, 1, 4] = 2.0
    targets = torch.tensor([[3, 4, IGNORE_INDEX, IGNORE_INDEX]])
    loss, info = parallel_semantic_loss(logits, targets, eos_token_id=2)
    loss.backward()
    assert info["valid_tokens"] == 2
    assert logits.grad[0, 2].abs().sum().item() == 0.0
    assert logits.grad[0, 3].abs().sum().item() == 0.0


def test_eos_target_and_decode_work():
    tokenizer = _Tokenizer()
    targets = encode_parallel_semantic_targets(tokenizer, ["a b"], max_tokens=4)
    assert targets.tolist() == [[3, 4, 2, IGNORE_INDEX]]
    text = decode_parallel_tokens(tokenizer, torch.tensor([[3, 4, 2, 7]]))
    assert text == ["3 4"]


def test_decode_parallel_tokens_truncates_at_first_eos_or_pad_before_batch_decode():
    tokenizer = _BatchDecodeDoesNotTruncateTokenizer()
    texts = decode_parallel_tokens(
        tokenizer,
        torch.tensor([
            [5, 6, 2, 7, 8],
            [9, 0, 10, 2, 11],
            [12, 13, 14, 15, 16],
        ]),
    )
    assert texts == ["5 6", "9", "12 13 14 15 16"]


def test_avg_pred_length_uses_first_eos_position():
    logits = torch.zeros(3, 5, 8)
    logits[0, :, 1] = 1.0
    logits[0, 2, 2] = 2.0  # first EOS at position 2
    logits[0, 4, 2] = 3.0
    logits[1, :, 1] = 1.0
    logits[1, 0, 2] = 2.0  # first EOS at position 0
    logits[2, :, 1] = 1.0  # no EOS, length K
    targets = torch.ones(3, 5, dtype=torch.long)
    _, info = parallel_semantic_loss(logits, targets, eos_token_id=2)
    assert abs(info["avg_pred_length"] - ((2 + 0 + 5) / 3)) < 1e-6


def test_freeze_all_except_parallel_readout():
    class _Experiment(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.parallel_readout = ParallelSemanticReadout(4, 8, num_queries=2, decoder_dim=4, num_heads=2)

    experiment = _Experiment()
    report = freeze_all_except_parallel_readout(experiment)
    assert report["trainable_parameters"] > 0
    assert all(not p.requires_grad for p in experiment.backbone.parameters())
    assert all(p.requires_grad for p in experiment.parallel_readout.parameters())


def test_semantic_backward_does_not_touch_original_model_params():
    backbone = nn.Linear(4, 4)
    readout = ParallelSemanticReadout(4, 8, num_queries=2, decoder_dim=4, num_heads=2)
    z = backbone(torch.randn(3, 4))
    logits = readout(z)
    targets = torch.tensor([[1, 2], [1, IGNORE_INDEX], [2, IGNORE_INDEX]])
    loss, _ = parallel_semantic_loss(logits, targets, eos_token_id=2)
    loss.backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in readout.parameters())


def test_anomaly_score_module_unchanged_by_attaching_readout():
    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_head = nn.Linear(4, 1)

        def score(self, x):
            return self.score_head(x)

    model = _Base()
    x = torch.randn(5, 4)
    before = model.score(x)
    model.parallel_readout = ParallelSemanticReadout(4, 8, num_queries=2, decoder_dim=4, num_heads=2)
    after = model.score(x)
    assert torch.equal(before, after)
