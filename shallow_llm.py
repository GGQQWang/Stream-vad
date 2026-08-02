"""Shallow LLM decoder — first K layers of Qwen2-VL + score head.

Used as the lightweight classifier on top of SSM window tokens.
The full LLM acts as teacher during training; only the first K layers
are used at inference time.  Layers are deep-copied so training the
student never modifies the teacher.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _get_llm_layers(model: nn.Module):
    """Resolve transformer layers across Qwen2-VL and LAVIDA variants.

    Qwen2-VL (HuggingFace):  ``model.language_model.layers``
    LavidaForCausalLM:       ``model.model.layers``
    """
    if hasattr(model.model, "language_model"):
        # vanilla Qwen2VLForConditionalGeneration
        lm = model.model.language_model
    elif hasattr(model.model, "layers"):
        # LavidaForCausalLM or other wrappers
        lm = model.model
    else:
        raise AttributeError(
            "Cannot find transformer layers on the supplied model. "
            "Expected .model.language_model.layers or .model.layers"
        )
    return lm.layers, lm.norm


class ShallowLLM(nn.Module):
    """First K transformer layers of Qwen2-VL + a scalar score head.

    Args:
        full_llm: loaded ``Qwen2VLForConditionalGeneration`` or
                  ``LavidaForCausalLM``.
        K: number of transformer layers to keep.  Default 4.
        llm_hidden: hidden dimension.  Default 3584 (Qwen2-VL-7B).
        text_head: if True, also output a text token logit head
                  (for anomaly descriptions).  Default False.

    Shape:
        Input:  ``[B, T, llm_hidden]``  SSM window tokens.
        Output: ``[B, T]``  scalar anomaly scores (+ optional logits).
    """

    def __init__(
        self,
        full_llm: nn.Module,
        K: int = 4,
        llm_hidden: int = 3584,
        text_head: bool = False,
    ):
        super().__init__()

        teacher_layers, teacher_norm = _get_llm_layers(full_llm)

        assert K <= len(teacher_layers), (
            f"Requested K={K} layers but LLM only has {len(teacher_layers)}"
        )

        # Deep-copy so student and teacher share no parameters
        self.layers = nn.ModuleList([
            copy.deepcopy(teacher_layers[i]) for i in range(K)
        ])
        self.norm = copy.deepcopy(teacher_norm)

        # Scalar anomaly score
        self.score_head = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden // 4),
            nn.GELU(),
            nn.Linear(llm_hidden // 4, 1),
        )

        # Optional text generation head
        self.text_head: Optional[nn.Linear] = None
        if text_head:
            self.text_head = nn.Linear(
                llm_hidden, full_llm.config.vocab_size
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: ``[B, T, llm_hidden]``  SSM tokens (already projected).

        Returns:
            scores: ``[B, T]``  anomaly scores.
            logits: ``[B, T, vocab_size]``  or ``None``.
        """
        B, T, _ = x.shape
        device = x.device

        # Position ids for full bidirectional attention within the
        # window sequence (SSM already handled causality).
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        h = x
        for layer in self.layers:
            out = layer(
                h,
                attention_mask=None,        # full bidirectional
                position_ids=position_ids,
            )
            # Decoder layer may return a tuple (hidden_states, ...) or
            # a plain tensor depending on the transformers version.
            h = out[0] if isinstance(out, tuple) else out

        h = self.norm(h)

        scores = self.score_head(h).squeeze(-1)  # [B, T]

        logits = None
        if self.text_head is not None:
            logits = self.text_head(h)           # [B, T, vocab_size]

        return scores, logits
