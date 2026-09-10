import pytest
import torch

import ibq_utils
from ibq_utils import (
    IBQTokenCache,
    build_codebook_metadata,
    build_ibq_cache_metadata,
    save_ibq_cache_atomic,
    validate_codebook_cache,
    validate_ibq_cache,
)


def _small_codebook(monkeypatch, size=8, dim=3):
    monkeypatch.setattr(ibq_utils, "IBQ_CODEBOOK_SIZE", size)
    monkeypatch.setattr(ibq_utils, "IBQ_CODE_EMBED_DIM", dim)
    return torch.arange(size * dim, dtype=torch.float32).reshape(size, dim)


def _write_codebook(cache_root, codebook, model_id="tok-a"):
    metadata = build_codebook_metadata(codebook=codebook, model_id=model_id)
    torch.save({"codebook": codebook, "metadata": metadata}, cache_root / "_codebook.pt")
    return metadata


def _write_video_cache(
    cache_root,
    *,
    video_id="v1",
    frames_per_clip=4,
    sample_interval=2,
    tokens_per_frame=5,
    model_id="tok-a",
    codebook_metadata,
    overrides=None,
):
    tokens = torch.zeros(2, frames_per_clip, tokens_per_frame, dtype=torch.int32)
    metadata = build_ibq_cache_metadata(
        video_id=video_id,
        n_windows=2,
        n_frames=16,
        fps=30.0,
        frames_per_clip=frames_per_clip,
        sample_interval=sample_interval,
        tokens_per_frame=tokens_per_frame,
        model_id=model_id,
        codebook_sha256=codebook_metadata["codebook_sha256"],
    )
    metadata.update(overrides or {})
    save_ibq_cache_atomic(cache_root, video_id=video_id, ibq_tokens=tokens, metadata=metadata)
    return metadata


def test_ibq_cache_validation_accepts_consistent_cache(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    validate_codebook_cache(tmp_path)
    _write_video_cache(tmp_path, codebook_metadata=codebook_metadata)

    validate_ibq_cache(
        tmp_path,
        video_id="v1",
        frames_per_clip=4,
        sample_interval=2,
        tokens_per_frame=5,
        codebook_metadata=codebook_metadata,
        n_windows=2,
        n_frames=16,
    )


def test_ibq_cache_validation_rejects_frames_per_clip_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(tmp_path, codebook_metadata=codebook_metadata)

    with pytest.raises(ValueError, match="frames_per_clip"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=8,
            sample_interval=2,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_cache_validation_rejects_sample_interval_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(tmp_path, codebook_metadata=codebook_metadata)

    with pytest.raises(ValueError, match="sample_interval"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=3,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_cache_validation_rejects_tokens_per_frame_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(tmp_path, codebook_metadata=codebook_metadata)

    with pytest.raises(ValueError, match="tokens_per_frame"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=2,
            tokens_per_frame=6,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_cache_validation_rejects_codebook_size_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(
        tmp_path,
        codebook_metadata=codebook_metadata,
        overrides={"codebook_size": codebook_metadata["codebook_size"] + 1},
    )

    with pytest.raises(ValueError, match="codebook_size"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=2,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_cache_validation_rejects_codebook_embed_dim_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(
        tmp_path,
        codebook_metadata=codebook_metadata,
        overrides={"codebook_embed_dim": codebook_metadata["codebook_embed_dim"] + 1},
    )

    with pytest.raises(ValueError, match="codebook_embed_dim"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=2,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_cache_validation_rejects_model_id_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(tmp_path, model_id="tok-b", codebook_metadata=codebook_metadata)

    with pytest.raises(ValueError, match="model_id"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=2,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_cache_validation_rejects_codebook_fingerprint_mismatch(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(
        tmp_path,
        codebook_metadata=codebook_metadata,
        overrides={"codebook_sha256": "not-the-current-codebook"},
    )

    with pytest.raises(ValueError, match="codebook_sha256"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=2,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_precompute_existing_cache_skip_uses_validation(tmp_path, monkeypatch):
    import precompute_ibq_tokens

    codebook = _small_codebook(monkeypatch)
    monkeypatch.setattr(precompute_ibq_tokens, "IBQ_TOKENS_PER_FRAME", 5)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(tmp_path, codebook_metadata=codebook_metadata)

    precompute_ibq_tokens._validate_existing_video_cache_for_skip(
        cache_root=tmp_path,
        video_id="v1",
        frames_per_clip=4,
        sample_interval=2,
        codebook_metadata=codebook_metadata,
        n_windows=2,
        n_frames=16,
    )


def test_precompute_existing_cache_skip_rejects_incompatible_cache(tmp_path, monkeypatch):
    import precompute_ibq_tokens

    codebook = _small_codebook(monkeypatch)
    monkeypatch.setattr(precompute_ibq_tokens, "IBQ_TOKENS_PER_FRAME", 5)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    _write_video_cache(tmp_path, codebook_metadata=codebook_metadata)

    with pytest.raises(ValueError, match="frames_per_clip"):
        precompute_ibq_tokens._validate_existing_video_cache_for_skip(
            cache_root=tmp_path,
            video_id="v1",
            frames_per_clip=8,
            sample_interval=2,
            codebook_metadata=codebook_metadata,
            n_windows=2,
            n_frames=16,
        )


def test_ibq_cache_validation_rejects_legacy_missing_fingerprint(tmp_path, monkeypatch):
    codebook = _small_codebook(monkeypatch)
    codebook_metadata = _write_codebook(tmp_path, codebook)
    legacy_metadata = {
        "video_id": "v1",
        "n_windows": 2,
        "n_frames": 16,
        "fps": 30.0,
        "frames_per_clip": 4,
        "sample_interval": 2,
        "tokens_per_frame": 5,
        "codebook_size": codebook_metadata["codebook_size"],
        "model_id": "tok-a",
    }
    torch.save(
        {
            "ibq_tokens": torch.zeros(2, 4, 5, dtype=torch.int32),
            "metadata": legacy_metadata,
        },
        tmp_path / "v1.pt",
    )

    with pytest.raises(ValueError, match="codebook_embed_dim"):
        validate_ibq_cache(
            tmp_path,
            video_id="v1",
            frames_per_clip=4,
            sample_interval=2,
            tokens_per_frame=5,
            codebook_metadata=codebook_metadata,
        )


def test_ibq_token_cache_rejects_padding_frame_from_metadata(tmp_path):
    cache_root = tmp_path / "ibq"
    cache_root.mkdir()
    tokens = torch.zeros(2, 16, 7, dtype=torch.int32)
    torch.save(
        {
            "ibq_tokens": tokens,
            "metadata": {
                "video_id": "v1",
                "n_windows": 2,
                "n_frames": 21,
                "fps": 30.0,
                "frames_per_clip": 16,
                "sample_interval": 1,
                "tokens_per_frame": 7,
            },
        },
        cache_root / "v1.pt",
    )

    cache = IBQTokenCache(cache_root)
    assert cache.valid_frame_count("v1", 1) == 5
    assert cache.get("v1", 1, 4).shape == (7,)
    with pytest.raises(IndexError):
        cache.get("v1", 1, 5)
