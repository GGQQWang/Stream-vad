"""Emu3.5 IBQ vision tokenizer wrapper and token-cache IO.

The tokenizer (BAAI/Emu3.5-VisionTokenizer) maps an image to a grid of
discrete codebook ids (16x spatial downsampling).  We use it as the
prediction target of the world-model auxiliary loss: the SSM hidden
state must predict which visual tokens appear in a future frame
("bag of tokens", one shared codebook distribution per window).

Cache layout (separate from the visual feature cache):
    {cache_root}/{video_id}.pt =
        {"metadata": {...}, "ibq_tokens": [n_windows, F, T] int32}
where F = frames per window (16), T = tokens per frame (392 for
448x224 input).
"""

from pathlib import Path
from typing import Optional

import torch

# CLIP normalization statistics (tokenizer pretraining preprocessing).
# The model itself performs no normalization; the caller must apply it.
IBQ_MEAN = (0.48145466, 0.4578275, 0.40821073)
IBQ_STD = (0.26862954, 0.26130258, 0.27577711)

# 448x224 = 100352 px, matching the visual pipeline's pixel budget.
# 16x downsampling gives a 28x14 = 392 token grid per frame.
IBQ_FRAME_SIZE = (448, 224)
IBQ_TOKENS_PER_FRAME = 392
IBQ_CODEBOOK_SIZE = 131072


def load_ibq_tokenizer(model_dir: str | Path, device, dtype=torch.float32):
    """Load Emu3p5VisionVQModel from a HF snapshot dir (local modeling files)."""
    from transformers import AutoConfig, AutoModel

    model_dir = str(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, config=config, trust_remote_code=True,
    ).to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def encode_frames(model, frames: torch.Tensor) -> torch.Tensor:
    """Encode frames to IBQ token ids.

    Args:
        model: Emu3p5VisionVQModel (or wrapper with ``encode``).
        frames: ``[B, 3, H, W]`` float in [0, 1].

    Returns:
        token_ids: ``[B, T]`` int64 codebook indices, flattened grid.
    """
    device = frames.device
    dtype = frames.dtype
    mean = torch.tensor(IBQ_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IBQ_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (frames - mean) / std
    # the tokenizer runs in its own dtype (e.g. bf16); cast the input
    x = x.to(dtype=model.dtype)
    _, _, (_, _, token_ids) = model.encode(x)
    # the tokenizer flattens the full [b, h, w] grid including the batch
    # dim; reshape back to per-frame rows
    return token_ids.reshape(frames.shape[0], -1)


def build_ibq_cache_metadata(
    *,
    video_id: str,
    n_windows: int,
    n_frames: int,
    fps: float,
    frames_per_clip: int,
    sample_interval: int,
    tokens_per_frame: int,
    model_id: str,
) -> dict:
    return {
        "video_id": video_id,
        "n_windows": int(n_windows),
        "n_frames": int(n_frames),
        "fps": float(fps),
        "frames_per_clip": int(frames_per_clip),
        "sample_interval": int(sample_interval),
        "tokens_per_frame": int(tokens_per_frame),
        "codebook_size": int(IBQ_CODEBOOK_SIZE),
        "model_id": str(model_id),
    }


def save_ibq_cache_atomic(
    cache_root: str | Path,
    *,
    video_id: str,
    ibq_tokens: torch.Tensor,
    metadata: dict,
) -> Path:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{video_id}.pt"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "ibq_tokens": ibq_tokens.detach().cpu().to(torch.int32),
        "metadata": metadata,
    }, tmp_path)
    tmp_path.replace(path)
    return path


def load_ibq_cache(cache_root: str | Path, *, video_id: str) -> dict:
    path = Path(cache_root) / f"{video_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"IBQ cache not found for {video_id}: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class IBQTokenCache:
    """Lazy per-video loader for IBQ token caches (training-side).

    Usage:
        cache = IBQTokenCache(root)
        tokens = cache.get(video_id, window_idx, frame_idx)  # [T] long
    """

    def __init__(self, cache_root: str | Path):
        self.root = Path(cache_root)
        self._loaded: dict = {}

    def get(self, video_id: str, window_idx: int, frame_idx: int) -> torch.Tensor:
        if video_id not in self._loaded:
            self._loaded[video_id] = load_ibq_cache(self.root, video_id=video_id)
        data = self._loaded[video_id]["ibq_tokens"]
        return data[window_idx, frame_idx]  # [T] int32

    @property
    def tokens_per_frame(self) -> int:
        # assume homogeneous cache; peek at the first loaded file or any file
        for video_id in self._loaded:
            return int(self._loaded[video_id]["metadata"]["tokens_per_frame"])
        raise RuntimeError("IBQTokenCache is empty; load a video first")
