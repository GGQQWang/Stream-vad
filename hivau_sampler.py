"""Per-video chunk sampler: shuffle video order, keep chunks sequential."""

import random
from typing import Iterator, List

from torch.utils.data import Sampler


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
