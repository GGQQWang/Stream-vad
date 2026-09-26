"""Exercise both vision-block APIs with tiny CPU visual towers."""

import torch
import torch.nn as nn

from vit_forwarder import ViTForwarder


class KeepAll(nn.Module):
    def forward(self, patches, grid_thw):
        # Reduction must see patch content, before Qwen3 learned positions.
        assert torch.all(patches == 0)
        return torch.ones(patches.shape[0], dtype=torch.bool), grid_thw[:, 1] * grid_thw[:, 2]


class OldBlock(nn.Module):
    def forward(self, x, cu_seqlens, rotary_pos_emb):
        assert rotary_pos_emb.shape == (4, 1)
        return x


class NewBlock(nn.Module):
    def forward(self, x, cu_seqlens, position_embeddings):
        cos, sin = position_embeddings
        assert cos.shape == sin.shape == (4, 2)
        return x


class Visual(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.blocks = nn.ModuleList([block])

    def patch_embed(self, values):
        return values

    def rot_pos_emb(self, grid_thw):
        return torch.zeros(4, 1)

    def merger(self, patches):
        return patches.mean(dim=0, keepdim=True).expand(1, 4096)


class Qwen3Visual(Visual):
    def fast_pos_embed_interpolate(self, grid_thw):
        return torch.ones(4, 4)


def test_qwen2_legacy_block_keeps_rotary_api():
    fwd = ViTForwarder(Visual(OldBlock()), KeepAll())
    tokens, counts = fwd.forward_batch(torch.zeros(4, 4), torch.tensor([[1, 2, 2]]))
    assert tokens.shape == (1, 4096)
    assert counts.tolist() == [1]


def test_qwen2_new_block_uses_cos_sin():
    fwd = ViTForwarder(Visual(NewBlock()), KeepAll())
    tokens, counts = fwd.forward_batch(torch.zeros(4, 4), torch.tensor([[1, 2, 2]]))
    assert tokens.shape == (1, 4096)
    assert counts.tolist() == [1]


def test_qwen3_adds_learned_positions_before_compression_batch_and_streaming():
    fwd = ViTForwarder(Qwen3Visual(NewBlock()), KeepAll())
    grid = torch.tensor([[1, 2, 2]])
    tokens, counts = fwd.forward_batch(torch.zeros(4, 4), grid)
    assert tokens.shape == (1, 4096)
    assert torch.all(tokens == 1)
    assert counts.tolist() == [1]
    streamed = fwd.forward_streaming(
        torch.zeros(4, 4), torch.tensor([0, 4]), grid, torch.arange(4),
    )
    assert streamed.shape == (1, 4096)
    assert torch.all(streamed == 1)
