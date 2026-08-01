"""Token compression and temporal modelling for streaming VAD.

Compression:
- TemporalTokenReducer: removes static patches across frames
- SpatialTokenCompressor: merges redundant visual tokens via LOF clustering

Streaming:
- StreamWindowManager: adaptive frame gating + patch buffer
- SSMBlock: Mamba2 window-to-window stateful temporal model
"""

from .temporal import TemporalTokenReducer
from .spatial import SpatialTokenCompressor
from .streaming import StreamWindowManager
from .ssm_block import SSMBlock

__all__ = [
    "TemporalTokenReducer",
    "SpatialTokenCompressor",
    "StreamWindowManager",
    "SSMBlock",
]
