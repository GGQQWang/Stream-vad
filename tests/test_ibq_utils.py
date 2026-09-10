import pytest
import torch

from ibq_utils import IBQTokenCache


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
