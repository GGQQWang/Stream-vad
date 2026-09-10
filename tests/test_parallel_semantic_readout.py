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

    def encode(self, text, add_special_tokens=False):
        table = {"a": 3, "b": 4}
        return [table[t] for t in text.split()]

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


def test_parallel_semantic_readout_shape():
    readout = ParallelSemanticReadout(8, 32, num_queries=16, decoder_dim=8, num_heads=2)
    logits = readout(torch.randn(3, 8))
    assert logits.shape == (3, 16, 32)


def test_parallel_queries_have_distinct_position_roles():
    readout = ParallelSemanticReadout(8, 32, num_queries=16, decoder_dim=8, num_heads=2)
    assert readout.query_embed.shape == (16, 8)
    assert torch.unique(readout.query_embed.detach(), dim=0).shape[0] == 16


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
    for p in backbone.parameters():
        p.requires_grad = False
    z = backbone(torch.randn(3, 4)).detach()
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
