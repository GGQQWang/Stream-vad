"""Shallow LLM decoder — first K layers of Qwen2-VL + score head.

Used as the lightweight classifier on top of SSM window tokens.
The full LLM acts as teacher during training; only the first K layers
are used at inference time.
"""

from typing import Optional

import torch
import torch.nn as nn


class ShallowLLM(nn.Module):
    """First K transformer layers of Qwen2-VL + a scalar score head.

    Args:
        full_llm: loaded ``Qwen2VLForConditionalGeneration`` (or
                  ``LavidaForCausalLM``).  Must have ``model.layers``
                  and ``model.norm``.
        K: number of transformer layers to keep.  Default 4.
        llm_hidden: hidden dimension of the LLM.  Default 3584
                   (Qwen2-VL-7B).
        text_head: if True, also output a text token logit head
                  (for anomaly descriptions).  Default False.

    Shape:
        Input:  ``[B, T, llm_hidden]``  SSM window tokens.
        Output: ``[B, T]``  scalar anomaly scores.
    """

    def __init__(
        self,
        full_llm: nn.Module,
        K: int = 4,
        llm_hidden: int = 3584,
        text_head: bool = False,
    ):
        super().__init__()

        # Slice first K transformer layers
        layers = full_llm.model.layers
        assert K <= len(layers), (
            f"Requested K={K} layers but LLM only has {len(layers)}"
        )
        self.layers = nn.ModuleList(layers[:K])
        self.norm = full_llm.model.norm

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
            x: ``[B, T, llm_hidden]``  SSM tokens.

        Returns:
            scores: ``[B, T]``  anomaly scores.
            logits: ``[B, T, vocab_size]``  if ``text_head`` enabled,
                   else ``None``.
        """
        h = x
        for layer in self.layers:
            h = layer(h)[0]                     # [B, T, llm_hidden]
        h = self.norm(h)

        scores = self.score_head(h).squeeze(-1)  # [B, T]

        logits = None
        if self.text_head is not None:
            logits = self.text_head(h)           # [B, T, vocab_size]

        return scores, logits
