"""Per-video samplers that keep chunks sequential within each video."""

import random
from typing import Iterator, List, Tuple

from torch.utils.data import Sampler

from mil_utils import cycle_pairs, group_video_chunks, split_normal_abnormal_videos


class VideoChunkSampler(Sampler):
    """Shuffle video order but keep chunks within each video in time order.

    Each chunk carries ``video_id`` and ``chunk_start``.  The sampler
    groups all chunks by ``video_id``, shuffles the video list, then
    yields indices in ``chunk_start`` order within each video.
    """

    def __init__(self, samples: List[dict], shuffle: bool = True):
        # samples  —  dataset.samples (list of per-chunk dicts)
        self.shuffle = shuffle

        # group chunk indices by video
        vid_to_chunks: dict = {}
        for i, s in enumerate(samples):
            vid = s["video_id"]
            vid_to_chunks.setdefault(vid, []).append((s["chunk_start"], i))

        # sort each video's chunks by chunk_start
        for vid in vid_to_chunks:
            vid_to_chunks[vid].sort(key=lambda x: x[0])

        self.video_ids = sorted(vid_to_chunks.keys())
        self.vid_chunks = vid_to_chunks

    def __len__(self) -> int:
        return sum(len(v) for v in self.vid_chunks.values())

    def __iter__(self) -> Iterator[int]:
        vid_order = list(self.video_ids)
        if self.shuffle:
            random.shuffle(vid_order)

        for vid in vid_order:
            for _, idx in self.vid_chunks[vid]:
                yield idx


class VideoPairSampler:
    """Yield one normal video and one abnormal video per MIL optimizer unit.

    The pair order may be shuffled, but each yielded video is represented by
    chunk indices sorted by ``chunk_start``.  If one class has fewer videos,
    it is cycled to match the larger class for that epoch.
    """

    def __init__(self, samples: List[dict], shuffle: bool = True, seed: int = 0):
        self.shuffle = shuffle
        self.seed = seed
        self.grouped = group_video_chunks(samples)
        self.normal_videos, self.abnormal_videos = split_normal_abnormal_videos(samples)
        if not self.normal_videos or not self.abnormal_videos:
            raise ValueError(
                "MIL training requires at least one normal and one abnormal video "
                f"(normal={len(self.normal_videos)}, abnormal={len(self.abnormal_videos)})"
            )
        self.num_pairs = max(len(self.normal_videos), len(self.abnormal_videos))

    def __len__(self) -> int:
        return self.num_pairs

    def iter_epoch(self, epoch: int = 0) -> Iterator[Tuple[str, List[int], str, List[int]]]:
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        if self.shuffle:
            pairs = cycle_pairs(
                self.normal_videos, self.abnormal_videos, generator=generator,
            )
        else:
            pairs = [
                (
                    self.normal_videos[i % len(self.normal_videos)],
                    self.abnormal_videos[i % len(self.abnormal_videos)],
                )
                for i in range(self.num_pairs)
            ]

        for normal_vid, abnormal_vid in pairs:
            normal_indices = [ref.index for ref in self.grouped[normal_vid]]
            abnormal_indices = [ref.index for ref in self.grouped[abnormal_vid]]
            yield normal_vid, normal_indices, abnormal_vid, abnormal_indices
