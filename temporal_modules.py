"""Interchangeable causal temporal modules for Stage-1 ablations."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_block import SSMBlock


TEMPORAL_MODELS = ("ssm", "lstm", "stc", "qformer", "transformer")
DEFAULT_TEMPORAL_HISTORY = 16


def detach_temporal_memory(memory):
    if memory is None:
        return None
    if isinstance(memory, torch.Tensor):
        return memory.detach()
    if isinstance(memory, dict):
        return {key: detach_temporal_memory(value) for key, value in memory.items()}
    if isinstance(memory, tuple):
        return tuple(detach_temporal_memory(value) for value in memory)
    if isinstance(memory, list):
        return [detach_temporal_memory(value) for value in memory]
    detach = getattr(memory, "detach", None)
    if callable(detach):
        return detach()
    raise TypeError(f"unsupported temporal memory type: {type(memory)!r}")


def temporal_memory_stats(memory) -> dict[str, int]:
    tensors: list[torch.Tensor] = []

    def collect(value) -> None:
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif hasattr(value, "conv_state") and hasattr(value, "ssm_state"):
            collect(value.conv_state)
            collect(value.ssm_state)
        else:
            raise TypeError(f"unsupported temporal memory type: {type(value)!r}")

    collect(memory)
    return {
        "num_elements": int(sum(tensor.numel() for tensor in tensors)),
        "bytes": int(sum(tensor.numel() * tensor.element_size() for tensor in tensors)),
    }


class CausalTemporalModule(nn.Module):
    temporal_model = "base"

    def __init__(self, d_input: int, d_state: int, output_dim: int, history_length: int):
        super().__init__()
        if d_input <= 0 or d_state <= 0 or output_dim <= 0:
            raise ValueError("temporal dimensions must be positive")
        if history_length <= 0:
            raise ValueError("temporal history must be positive")
        self.d_input = int(d_input)
        self.state_dim = int(d_state)
        self.output_dim = int(output_dim)
        self.history_length = int(history_length)
        self.in_proj = nn.Sequential(
            nn.Linear(self.d_input, self.state_dim),
            nn.LayerNorm(self.state_dim),
        )
        self.out_norm = nn.LayerNorm(self.state_dim)
        self.out_proj = nn.Linear(self.state_dim, self.output_dim)
        self.debug_device = False

    def step(self, x_t: torch.Tensor, memory=None):
        raise NotImplementedError

    def forward_sequence(self, x: torch.Tensor, memory=None):
        if x.ndim != 3:
            raise ValueError(f"temporal input must be [B, T, D], got {tuple(x.shape)}")
        states = []
        current_memory = memory
        for index in range(x.shape[1]):
            state_t, current_memory = self.step(x[:, index], current_memory)
            states.append(state_t)
        if not states:
            empty = x.new_empty(x.shape[0], 0, self.state_dim)
            return empty, current_memory
        return torch.stack(states, dim=1), current_memory

    def forward_chunk(self, x: torch.Tensor, state=None, return_internal: bool = False):
        internal, new_state = self.forward_sequence(x, state)
        output = self.out_proj(internal)
        if return_internal:
            return output, new_state, internal
        return output, new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_chunk(x)
        return output

    def reset_state(self):
        return None

    def get_memory_stats(self, memory) -> dict[str, int]:
        return temporal_memory_stats(memory)

    def get_config(self) -> dict:
        return {
            "name": self.temporal_model,
            "d_input": self.d_input,
            "state_dim": self.state_dim,
            "output_dim": self.output_dim,
            "history_length": self.history_length,
        }


class LSTMTemporalModule(CausalTemporalModule):
    temporal_model = "lstm"

    def __init__(self, d_input: int, d_state: int, output_dim: int, history_length: int):
        super().__init__(d_input, d_state, output_dim, history_length)
        self.cell = nn.LSTMCell(self.state_dim, self.state_dim)
        self.num_layers = 1

    def step(self, x_t: torch.Tensor, memory=None):
        projected = self.in_proj(x_t)
        if memory is None:
            h_prev = projected.new_zeros(projected.shape[0], self.state_dim)
            c_prev = projected.new_zeros(projected.shape[0], self.state_dim)
        else:
            h_prev, c_prev = memory
            h_prev = h_prev.to(device=projected.device, dtype=projected.dtype)
            c_prev = c_prev.to(device=projected.device, dtype=projected.dtype)
        h_t, c_t = self.cell(projected, (h_prev, c_prev))
        return self.out_norm(h_t), (h_t, c_t)

    def get_config(self) -> dict:
        return {**super().get_config(), "num_layers": self.num_layers}


class _HistoryTemporalModule(CausalTemporalModule):
    def _append_history(self, projected: torch.Tensor, memory) -> torch.Tensor:
        current = projected.unsqueeze(1)
        if memory is None:
            history = current
        else:
            memory = memory.to(device=projected.device, dtype=projected.dtype)
            if memory.ndim != 3 or memory.shape[0] != projected.shape[0]:
                raise ValueError("temporal history must be [B, T, d_state] with matching batch size")
            history = torch.cat([memory, current], dim=1)
        return history[:, -self.history_length:]


class STCTemporalModule(_HistoryTemporalModule):
    """Two-block causal temporal convolution over the visible history."""

    temporal_model = "stc"

    def __init__(self, d_input: int, d_state: int, output_dim: int, history_length: int):
        super().__init__(d_input, d_state, output_dim, history_length)
        self.pre_norm = nn.LayerNorm(self.state_dim)
        self.conv1 = nn.Conv1d(self.state_dim, self.state_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(self.state_dim, self.state_dim, kernel_size=3)
        self.num_layers = 2

    @staticmethod
    def _causal_conv(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, (conv.kernel_size[0] - 1, 0)))

    def step(self, x_t: torch.Tensor, memory=None):
        projected = self.in_proj(x_t)
        history = self._append_history(projected, memory)
        sequence = self.pre_norm(history).transpose(1, 2)
        sequence = F.gelu(self._causal_conv(self.conv1, sequence))
        sequence = self._causal_conv(self.conv2, sequence).transpose(1, 2)
        state_t = self.out_norm(history[:, -1] + sequence[:, -1])
        return state_t, history

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "kernel_size": 3,
        }


def _choose_nhead(d_state: int) -> int:
    for nhead in (8, 4, 2, 1):
        if d_state % nhead == 0:
            return nhead
    return 1


class TransformerTemporalModule(_HistoryTemporalModule):
    temporal_model = "transformer"

    def __init__(self, d_input: int, d_state: int, output_dim: int, history_length: int):
        super().__init__(d_input, d_state, output_dim, history_length)
        self.num_layers = 2
        self.nhead = _choose_nhead(self.state_dim)
        self.position = nn.Parameter(torch.randn(history_length, self.state_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.state_dim,
            nhead=self.nhead,
            dim_feedforward=4 * self.state_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)

    def step(self, x_t: torch.Tensor, memory=None):
        projected = self.in_proj(x_t)
        history = self._append_history(projected, memory)
        length = history.shape[1]
        positioned = history + self.position[:length].to(history.dtype).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(length, length, device=history.device, dtype=torch.bool), diagonal=1,
        )
        encoded = self.encoder(positioned, mask=causal_mask)
        return self.out_norm(encoded[:, -1]), history

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "nhead": self.nhead,
            "feedforward_dim": 4 * self.state_dim,
            "dropout": 0.1,
        }


class _QFormerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, queries: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        normed = self.query_norm(queries)
        attended, _ = self.self_attn(normed, normed, normed, need_weights=False)
        queries = queries + attended
        attended, _ = self.cross_attn(
            self.cross_norm(queries), self.memory_norm(history), self.memory_norm(history),
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class QFormerTemporalModule(_HistoryTemporalModule):
    temporal_model = "qformer"

    def __init__(self, d_input: int, d_state: int, output_dim: int, history_length: int):
        super().__init__(d_input, d_state, output_dim, history_length)
        self.num_layers = 2
        self.num_queries = 4
        self.nhead = _choose_nhead(self.state_dim)
        self.queries = nn.Parameter(torch.randn(self.num_queries, self.state_dim) * 0.02)
        self.position = nn.Parameter(torch.randn(history_length, self.state_dim) * 0.02)
        self.layers = nn.ModuleList([
            _QFormerLayer(self.state_dim, self.nhead) for _ in range(self.num_layers)
        ])

    def step(self, x_t: torch.Tensor, memory=None):
        projected = self.in_proj(x_t)
        history = self._append_history(projected, memory)
        length = history.shape[1]
        visible_history = history + self.position[:length].to(history.dtype).unsqueeze(0)
        queries = self.queries.to(history.dtype).unsqueeze(0).expand(history.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, visible_history)
        return self.out_norm(queries.mean(dim=1)), history

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_layers": self.num_layers,
            "num_queries": self.num_queries,
            "nhead": self.nhead,
            "dropout": 0.1,
        }


def build_temporal_module(
    name: str,
    *,
    d_input: int,
    d_state: int,
    output_dim: int,
    history_length: int = DEFAULT_TEMPORAL_HISTORY,
    ssm_layers: int = 1,
    ssm_cls: Callable[..., nn.Module] | None = None,
    match_ssm_rng: bool = False,
) -> nn.Module:
    name = str(name).lower()
    if name not in TEMPORAL_MODELS:
        raise ValueError(f"unknown temporal model {name!r}; expected one of {TEMPORAL_MODELS}")
    if history_length <= 0:
        raise ValueError("temporal history must be positive")
    if name == "ssm":
        cls = SSMBlock if ssm_cls is None else ssm_cls
        return cls(d_input=d_input, d_model=d_state, n_layers=ssm_layers, llm_hidden=output_dim)
    classes = {
        "lstm": LSTMTemporalModule,
        "stc": STCTemporalModule,
        "qformer": QFormerTemporalModule,
        "transformer": TransformerTemporalModule,
    }
    if not match_ssm_rng:
        return classes[name](d_input, d_state, output_dim, history_length)

    # Keep every non-temporal parameter initialized identically to the SSM
    # reference under the same seed.  The baseline is initialized from the
    # current RNG state, while global RNG advances exactly as legacy SSM
    # construction would have advanced it before adapter/heads/queries.
    cls = SSMBlock if ssm_cls is None else ssm_cls
    rng_before = torch.random.get_rng_state()
    reference = cls(
        d_input=d_input, d_model=d_state, n_layers=ssm_layers, llm_hidden=output_dim,
    )
    rng_after_reference = torch.random.get_rng_state()
    del reference
    torch.random.set_rng_state(rng_before)
    module = classes[name](d_input, d_state, output_dim, history_length)
    torch.random.set_rng_state(rng_after_reference)
    return module


def checkpoint_temporal_config(
    checkpoint: dict,
    *,
    requested_model: str | None = None,
    requested_history: int | None = None,
    warn: Callable[[str], None] = print,
) -> tuple[str, int]:
    saved_model = checkpoint.get("temporal_model")
    if saved_model is None:
        saved_model = "ssm"
        warn("WARNING: checkpoint has no temporal_model metadata; treating it as legacy ssm.")
    saved_model = str(saved_model).lower()
    if saved_model not in TEMPORAL_MODELS:
        raise ValueError(f"checkpoint has unknown temporal_model={saved_model!r}")
    config = checkpoint.get("temporal_config") or {}
    saved_history = int(checkpoint.get(
        "temporal_history", config.get("history_length", DEFAULT_TEMPORAL_HISTORY),
    ))
    if saved_history <= 0:
        raise ValueError(f"checkpoint has invalid temporal_history={saved_history}")
    if requested_model is not None:
        requested_model = str(requested_model).lower()
    if requested_model is not None and saved_model != requested_model:
        raise ValueError(
            f"temporal model mismatch: checkpoint={saved_model!r}, requested={requested_model!r}"
        )
    if requested_history is not None and saved_model in {"stc", "qformer", "transformer"}:
        if saved_history != int(requested_history):
            raise ValueError(
                f"temporal history mismatch: checkpoint={saved_history}, requested={requested_history}"
            )
    return saved_model, saved_history
