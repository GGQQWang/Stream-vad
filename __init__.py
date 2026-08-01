"""Token compression and temporal modelling for streaming VAD.

Compression:
- TemporalTokenReducer: removes static patches across frames
- SpatialTokenCompressor: merges redundant visual tokens via LOF clustering

Streaming:
- StreamWindowManager: adaptive frame gating + patch buffer
- SSMBlock: Mamba2 window-to-window stateful temporal model

Integration:
- ViTForwarder: Qwen2-VL ViT blocks + merger (batch & streaming)
- ShallowLLM: first K layers of Qwen2-VL + score head
"""

from .temporal import TemporalTokenReducer
from .spatial import SpatialTokenCompressor
from .streaming import StreamWindowManager
from .ssm_block import SSMBlock
from .vit_forwarder import ViTForwarder
from .shallow_llm import ShallowLLM

__all__ = [
    "TemporalTokenReducer",
    "SpatialTokenCompressor",
    "StreamWindowManager",
    "SSMBlock",
    "ViTForwarder",
    "ShallowLLM",
]
