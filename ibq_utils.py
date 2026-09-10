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

import hashlib
from pathlib import Path
from typing import Optional

import torch

from stage1_streaming import sampled_frame_count_for_window

# CLIP normalization statistics (tokenizer pretraining preprocessing).
# The model itself performs no normalization; the caller must apply it.
IBQ_MEAN = (0.48145466, 0.4578275, 0.40821073)
IBQ_STD = (0.26862954, 0.26130258, 0.27577711)

# 448x224 = 100352 px, matching the visual pipeline's pixel budget.
# 16x downsampling gives a 28x14 = 392 token grid per frame.
IBQ_FRAME_SIZE = (448, 224)
IBQ_TOKENS_PER_FRAME = 392
IBQ_CODEBOOK_SIZE = 131072
IBQ_CODE_EMBED_DIM = 256
IBQ_CACHE_VERSION = 2
IBQ_CODEBOOK_CACHE_VERSION = 1


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.float().numpy().tobytes())
    return digest.hexdigest()


def build_codebook_metadata(*, codebook: torch.Tensor, model_id: str) -> dict:
    if codebook.ndim != 2:
        raise ValueError(f"IBQ codebook must be 2D, got shape {tuple(codebook.shape)}")
    return {
        "cache_version": IBQ_CODEBOOK_CACHE_VERSION,
        "model_id": str(model_id),
        "codebook_size": int(codebook.shape[0]),
        "codebook_embed_dim": int(codebook.shape[1]),
        "codebook_dtype": str(codebook.dtype),
        "codebook_sha256": tensor_sha256(codebook),
    }


def load_codebook_cache(cache_root: str | Path) -> dict:
    path = Path(cache_root) / "_codebook.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"IBQ codebook not found at {path}; re-run precompute_ibq_tokens.py "
            "once (it saves the codebook automatically)"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        payload = {"codebook": payload}
    if "codebook" not in payload:
        raise ValueError(f"invalid IBQ codebook cache at {path}: missing codebook")
    return payload


def validate_codebook_cache(
    cache_root: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_codebook: torch.Tensor | None = None,
) -> dict:
    path = Path(cache_root) / "_codebook.pt"
    payload = load_codebook_cache(cache_root)
    codebook = payload["codebook"]
    if codebook.ndim != 2:
        raise ValueError(f"IBQ codebook cache mismatch at {path}: codebook must be 2D, got {tuple(codebook.shape)}")
    expected_shape = (IBQ_CODEBOOK_SIZE, IBQ_CODE_EMBED_DIM)
    if tuple(codebook.shape) != expected_shape:
        raise ValueError(
            f"IBQ codebook cache mismatch at {path}: codebook shape expected "
            f"{expected_shape}, got {tuple(codebook.shape)}"
        )
    actual_sha256 = tensor_sha256(codebook)
    if expected_codebook is not None:
        expected_sha256 = tensor_sha256(expected_codebook)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"IBQ codebook cache mismatch at {path}: codebook_sha256 expected "
                f"{expected_sha256}, got {actual_sha256}"
            )

    metadata = dict(payload.get("metadata", {}))
    if metadata:
        checks = {
            "codebook_size": IBQ_CODEBOOK_SIZE,
            "codebook_embed_dim": IBQ_CODE_EMBED_DIM,
            "codebook_sha256": actual_sha256,
        }
        if expected_model_id is not None:
            checks["model_id"] = str(expected_model_id)
        for key, expected in checks.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"IBQ codebook cache mismatch at {path}: {key} expected "
                    f"{expected!r}, got {actual!r}"
                )
    elif expected_codebook is None:
        # A legacy _codebook.pt without metadata can still be tied to video
        # caches by the computed fingerprint, but it cannot carry model_id.
        metadata = {}

    metadata.setdefault("codebook_size", int(codebook.shape[0]))
    metadata.setdefault("codebook_embed_dim", int(codebook.shape[1]))
    metadata.setdefault("codebook_sha256", actual_sha256)
    return metadata


def save_codebook(cache_root: str | Path, model, model_id: str = "") -> Path:
    """Persist the tokenizer's (frozen) codebook into the cache root.

    Training computes codebook logits as a dot product against this
    frozen embedding instead of learning a huge [V, H] output layer.
    """
    path = Path(cache_root) / "_codebook.pt"
    weight = model.quantize.embedding.weight.detach().cpu()
    if path.is_file():
        validate_codebook_cache(cache_root, expected_model_id=model_id, expected_codebook=weight)
        return path
    metadata = build_codebook_metadata(codebook=weight, model_id=model_id)
    torch.save({"codebook": weight, "metadata": metadata}, path)
    return path


def load_codebook(cache_root: str | Path) -> torch.Tensor:
    return load_codebook_cache(cache_root)["codebook"]


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
    codebook_sha256: str = "",
) -> dict:
    return {
        "cache_version": IBQ_CACHE_VERSION,
        "video_id": video_id,
        "n_windows": int(n_windows),
        "n_frames": int(n_frames),
        "fps": float(fps),
        "frames_per_clip": int(frames_per_clip),
        "sample_interval": int(sample_interval),
        "tokens_per_frame": int(tokens_per_frame),
        "codebook_size": int(IBQ_CODEBOOK_SIZE),
        "codebook_embed_dim": int(IBQ_CODE_EMBED_DIM),
        "codebook_sha256": str(codebook_sha256),
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


def load_ibq_cache_header(cache_root: str | Path, *, video_id: str) -> dict:
    path = Path(cache_root) / f"{video_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"IBQ cache not found for {video_id}: {path}")
    try:
        return torch.load(path, map_location="meta", weights_only=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def validate_ibq_cache(
    cache_root: str | Path,
    *,
    video_id: str,
    frames_per_clip: int,
    sample_interval: int,
    tokens_per_frame: int,
    codebook_metadata: dict,
    n_windows: int | None = None,
    n_frames: int | None = None,
) -> dict:
    path = Path(cache_root) / f"{video_id}.pt"
    payload = load_ibq_cache_header(cache_root, video_id=video_id)
    if "metadata" not in payload or "ibq_tokens" not in payload:
        raise ValueError(f"invalid IBQ cache for video {video_id} at {path}: missing metadata or ibq_tokens")
    metadata = payload["metadata"]
    tokens = payload["ibq_tokens"]
    if tokens.ndim != 3:
        raise ValueError(
            f"IBQ cache mismatch for video {video_id} at {path}: ibq_tokens "
            f"must be 3D, got {tuple(tokens.shape)}"
        )

    expected = {
        "video_id": str(video_id),
        "frames_per_clip": int(frames_per_clip),
        "sample_interval": int(sample_interval),
        "tokens_per_frame": int(tokens_per_frame),
        "codebook_size": int(codebook_metadata["codebook_size"]),
        "codebook_embed_dim": int(codebook_metadata["codebook_embed_dim"]),
        "codebook_sha256": str(codebook_metadata["codebook_sha256"]),
    }
    if codebook_metadata.get("model_id"):
        expected["model_id"] = str(codebook_metadata["model_id"])
    if n_windows is not None:
        expected["n_windows"] = int(n_windows)
    if n_frames is not None:
        expected["n_frames"] = int(n_frames)

    for key, expected_value in expected.items():
        if key not in metadata:
            raise ValueError(
                f"IBQ cache mismatch for video {video_id} at {path}: missing "
                f"metadata field {key}"
            )
        actual = metadata[key]
        actual_value = int(actual) if isinstance(expected_value, int) else str(actual)
        if actual_value != expected_value:
            raise ValueError(
                f"IBQ cache mismatch for video {video_id} at {path}: {key} "
                f"expected {expected_value!r}, got {actual!r}"
            )

    expected_shape = (
        int(metadata["n_windows"]),
        int(metadata["frames_per_clip"]),
        int(metadata["tokens_per_frame"]),
    )
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(
            f"IBQ cache mismatch for video {video_id} at {path}: ibq_tokens "
            f"shape expected {expected_shape}, got {tuple(tokens.shape)}"
        )
    return metadata


def validate_ibq_cache_set(
    cache_root: str | Path,
    *,
    video_ids: list[str],
    frames_per_clip: int,
    sample_interval: int,
    tokens_per_frame: int,
    codebook_metadata: dict,
    expected_by_video: dict[str, dict] | None = None,
) -> None:
    for video_id in sorted(set(video_ids)):
        expected_video = (expected_by_video or {}).get(video_id, {})
        validate_ibq_cache(
            cache_root,
            video_id=video_id,
            frames_per_clip=frames_per_clip,
            sample_interval=sample_interval,
            tokens_per_frame=tokens_per_frame,
            codebook_metadata=codebook_metadata,
            n_windows=expected_video.get("n_windows"),
            n_frames=expected_video.get("n_frames"),
        )


class IBQTokenCache:
    """Lazy per-video loader for IBQ token caches (training-side).

    Usage:
        cache = IBQTokenCache(root)
        tokens = cache.get(video_id, window_idx, frame_idx)  # [T] long
    """

    def __init__(self, cache_root: str | Path):
        self.root = Path(cache_root)
        self._loaded: dict = {}
        self._headers: dict = {}

    def _header(self, video_id: str) -> dict:
        if video_id in self._loaded:
            return self._loaded[video_id]
        if video_id not in self._headers:
            self._headers[video_id] = load_ibq_cache_header(self.root, video_id=video_id)
        return self._headers[video_id]

    def num_windows(self, video_id: str) -> int:
        header = self._header(video_id)
        if "metadata" not in header or "ibq_tokens" not in header:
            raise ValueError(
                f"invalid IBQ cache for {video_id}: missing metadata or ibq_tokens"
            )
        data = header["ibq_tokens"]
        metadata = header["metadata"]
        if "n_windows" not in metadata:
            raise ValueError(
                f"IBQ cache metadata for {video_id} is missing n_windows"
            )
        n_windows = int(metadata["n_windows"])
        if n_windows != int(data.shape[0]):
            raise ValueError(
                f"IBQ cache shape mismatch for {video_id}: n_windows metadata "
                f"{n_windows} but ibq_tokens stores {data.shape[0]}"
            )
        return n_windows

    def get(self, video_id: str, window_idx: int, frame_idx: int) -> torch.Tensor:
        if video_id not in self._loaded:
            self._loaded[video_id] = load_ibq_cache(self.root, video_id=video_id)
        cache = self._loaded[video_id]
        data = cache["ibq_tokens"]
        if window_idx < 0 or window_idx >= int(data.shape[0]):
            raise IndexError(f"{video_id}: window_idx={window_idx} outside IBQ cache")
        valid_frames = self.valid_frame_count(video_id, window_idx)
        if frame_idx < 0 or frame_idx >= valid_frames:
            raise IndexError(
                f"{video_id}: frame_idx={frame_idx} outside valid sampled frames "
                f"for window_idx={window_idx} (valid={valid_frames})"
            )
        return data[window_idx, frame_idx]  # [T] int32

    def valid_frame_count(self, video_id: str, window_idx: int) -> int:
        if video_id not in self._loaded:
            self._loaded[video_id] = load_ibq_cache(self.root, video_id=video_id)
        cache = self._loaded[video_id]
        data = cache["ibq_tokens"]
        if window_idx < 0 or window_idx >= int(data.shape[0]):
            raise IndexError(f"{video_id}: window_idx={window_idx} outside IBQ cache")
        metadata = cache.get("metadata", {})
        required = ("n_frames", "frames_per_clip", "sample_interval")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(
                f"IBQ cache metadata for {video_id} is missing {missing}; "
                "cannot distinguish valid frames from padding"
            )
        valid_frames = sampled_frame_count_for_window(
            n_frames=int(metadata["n_frames"]),
            window_index=int(window_idx),
            frames_per_clip=int(metadata["frames_per_clip"]),
            sample_interval=int(metadata["sample_interval"]),
        )
        if valid_frames <= 0:
            raise IndexError(
                f"{video_id}: window_idx={window_idx} has no valid sampled frames"
            )
        if valid_frames > int(data.shape[1]):
            raise ValueError(
                f"IBQ cache shape mismatch for {video_id}: window_idx={window_idx} "
                f"has {valid_frames} valid frames but cache stores {data.shape[1]}"
            )
        return valid_frames

    @property
    def tokens_per_frame(self) -> int:
        # assume homogeneous cache; peek at the first loaded file or any file
        for video_id in self._loaded:
            return int(self._loaded[video_id]["metadata"]["tokens_per_frame"])
        raise RuntimeError("IBQTokenCache is empty; load a video first")
