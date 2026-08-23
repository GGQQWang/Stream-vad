"""Precompute IBQ tokens for world-model training targets.

For every scoring window of every training video, encode ALL sampled
frames (16 per window) with the frozen Emu3.5 IBQ tokenizer and store
the codebook ids in a separate cache dir.  Training then samples one
random frame per window per step as the CE target.

Windows are built with the exact same logic as training
(stage1_streaming.build_window_infos) so the cache aligns with the
visual feature cache window-for-window.  Only the sampled frames are
decoded (not the whole video), resizing runs on GPU, and the tokenizer
runs in bf16.

Run (server, ~5-8h):
    python -u precompute_ibq_tokens.py \
        --annotation-json /data3/wgq/data/HIVAU-70k/raw_annotations/ucf_database_train.json \
        --video-root /data1/wjq/data/UCF_Crime/training/videos \
        --ibq-model-dir /data3/wgq/models/Emu3.5-VisionTokenizer \
        --cache-root /data3/wgq/data/ucf_ibq_cache_100352 \
        --frames-per-clip 16 --sample-interval 3 \
        --device cuda:N
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ibq_utils import (
    IBQ_FRAME_SIZE,
    build_ibq_cache_metadata,
    encode_frames,
    load_ibq_tokenizer,
    save_codebook,
    save_ibq_cache_atomic,
)
from stage1_streaming import build_window_infos


def _decode_video_frames(video_path: str | Path, needed_indices):
    """Decode only the requested frame indices.

    Returns ``(frames [N, 3, H, W] float in [0,1], index_list)`` where
    ``frames[i]`` is source frame ``index_list[i]``.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    indices = sorted({max(0, min(int(i), total - 1)) for i in needed_indices})
    arr = vr.get_batch(indices).asnumpy()                    # [N, H, W, 3]
    frames = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return frames, indices


@torch.no_grad()
def _encode_frames_gpu(model, frames: torch.Tensor, device: torch.device, batch: int):
    """Resize (on GPU) + IBQ-encode frames in batches."""
    H, W = IBQ_FRAME_SIZE
    parts = []
    for start in range(0, frames.shape[0], batch):
        chunk = frames[start:start + batch].to(device)
        chunk = F.interpolate(chunk, size=(H, W), mode="bilinear", align_corners=False)
        parts.append(encode_frames(model, chunk).cpu())
    return torch.cat(parts, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--ibq-model-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--sample-interval", type=int, default=3)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--debug-video", default="",
                       help="optional single video id to sanity-check")
    args = parser.parse_args()

    import json

    device = torch.device(args.device)
    with open(args.annotation_json, "r") as f:
        raw = json.load(f)

    print("Loading IBQ tokenizer ...")
    model = load_ibq_tokenizer(args.ibq_model_dir, device, dtype=torch.bfloat16)
    model_id = args.model_id or str(args.ibq_model_dir)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    # persist the frozen codebook once so training can compute dot-product
    # logits against it without loading the tokenizer
    save_codebook(cache_root, model)

    n_done = 0
    n_skip = 0
    n_missing = 0
    for video_id, meta in tqdm(raw.items(), desc="Precompute IBQ"):
        if args.debug_video and video_id != args.debug_video:
            continue
        n_frames = int(meta["n_frames"])
        fps = float(meta.get("fps", 30.0))
        out_path = cache_root / f"{video_id}.pt"
        if out_path.is_file():
            n_skip += 1
            continue

        video_path = Path(args.video_root) / f"{video_id}.mp4"
        if not video_path.exists():
            n_missing += 1
            continue

        # same windowing as training
        infos = build_window_infos(
            n_frames=n_frames,
            fps=fps,
            anomaly_intervals=[],
            frames_per_clip=args.frames_per_clip,
            sample_interval=args.sample_interval,
            summary_clips=None,
        )
        n_windows = len(infos)
        if n_windows == 0:
            continue

        # decode only the sampled frames (not the whole video)
        needed = [fi for info in infos for fi in info.sampled_frames]
        frames, index_list = _decode_video_frames(video_path, needed)
        row_of = {fi: i for i, fi in enumerate(index_list)}

        ids = _encode_frames_gpu(model, frames, device, args.encode_batch)  # [N, T]
        tokens_per_frame = int(ids.shape[1])
        F = args.frames_per_clip
        ibq_tokens = torch.zeros(n_windows, F, tokens_per_frame, dtype=torch.int32)
        for wi, info in enumerate(infos):
            for fi, frame_idx in enumerate(info.sampled_frames):
                ibq_tokens[wi, fi] = ids[row_of[frame_idx]].to(torch.int32)

        # sanity: codebook usage diversity (degenerate = wrong normalization)
        if n_done == 0:
            uniq = torch.unique(ibq_tokens.reshape(-1))
            usage = float(uniq.numel()) / float(ibq_tokens.numel())
            print(f"[sanity] {video_id}: unique_tokens={uniq.numel()} "
                  f"usage_ratio={usage:.4f} (very low => normalization may be wrong)")

        metadata = build_ibq_cache_metadata(
            video_id=video_id,
            n_windows=n_windows,
            n_frames=n_frames,
            fps=fps,
            frames_per_clip=args.frames_per_clip,
            sample_interval=args.sample_interval,
            tokens_per_frame=tokens_per_frame,
            model_id=model_id,
        )
        save_ibq_cache_atomic(cache_root, video_id=video_id, ibq_tokens=ibq_tokens, metadata=metadata)
        n_done += 1

    print(f"IBQ precompute done: wrote={n_done}, skipped={n_skip}, missing_videos={n_missing}")


if __name__ == "__main__":
    main()
