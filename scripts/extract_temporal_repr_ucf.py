"""Extract final-layer SCORE hidden states once for offline temporal analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hivau_dataset import HIVAUDataset  # noqa: E402
from infer_stage1_ucf import (  # noqa: E402
    FRAMES_PER_CLIP,
    MAX_PIXELS,
    MAX_WINDOWS,
    MIN_PIXELS,
    SAMPLE_INTERVAL,
    SPATIAL_FUSION_MODES,
    infer_video,
    load_stage1_model,
    normalize_manifest,
)
from mil_utils import group_video_chunks  # noqa: E402
from temporal_modules import TEMPORAL_MODELS  # noqa: E402
from temporal_representation import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    load_repr_video,
    save_json,
    save_repr_video,
)


def _cache_is_complete(output_dir: Path, args) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if (
        int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("method") != args.method
    ):
        return False
    expected_paths = {
        "stage1_dir": args.stage1_dir,
        "model_path": args.model_path,
        "test_manifest": args.test_manifest,
        "feature_cache_root": args.feature_cache_root,
        "gt_root": args.gt_root,
    }
    for key, value in expected_paths.items():
        if Path(str(manifest.get(key, ""))).resolve() != Path(value).resolve():
            return False
    if manifest.get("storage_dtype") != args.storage_dtype:
        return False
    entries = manifest.get("videos", [])
    if not entries or int(manifest.get("num_videos", -1)) != len(entries):
        return False
    for entry in entries:
        path = output_dir / entry["file"]
        if not path.is_file():
            return False
        cached = load_repr_video(path)
        if cached["method"] != args.method or cached["video_id"] != entry["video_id"]:
            return False
    return True


def _existing_video(path: Path, method: str, video_id: str) -> dict | None:
    if not path.is_file():
        return None
    cached = load_repr_video(path)
    if cached["method"] != method:
        raise ValueError(f"{path}: cache method={cached['method']!r}, expected {method!r}")
    if cached["video_id"] != video_id:
        raise ValueError(f"{path}: cache contains video_id={cached['video_id']!r}, expected {video_id!r}")
    return cached


def _cache_protocol(args) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": args.method,
        "stage1_dir": str(Path(args.stage1_dir).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "feature_cache_root": str(Path(args.feature_cache_root).resolve()),
        "gt_root": str(Path(args.gt_root).resolve()),
        "storage_dtype": args.storage_dtype,
        "representation": "final_layer_SCORE_hidden_input_to_score_head",
        "interval": "frame_[start_frame,valid_end_frame)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=TEMPORAL_MODELS, required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and _cache_is_complete(output_dir, args) and not args.overwrite:
        print(f"Representation cache is complete; leaving it unchanged: {output_dir}")
        return
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output directory is non-empty but incomplete: {output_dir}; "
            "pass --resume or --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    protocol = _cache_protocol(args)
    if args.resume:
        existing_npz = list(output_dir.glob("*.npz"))
        if existing_npz and not manifest_path.is_file():
            raise ValueError(
                "cannot verify checkpoint provenance for NPZ files without manifest.json; "
                "use --overwrite"
            )
        if manifest_path.is_file():
            with open(manifest_path) as handle:
                saved_manifest = json.load(handle)
            mismatches = {
                key: (saved_manifest.get(key), value)
                for key, value in protocol.items()
                if saved_manifest.get(key) != value
            }
            if mismatches:
                raise ValueError(f"representation cache protocol mismatch: {mismatches}")

    args.device = torch.device(args.device)
    normalized_manifest = normalize_manifest(args.test_manifest, output_dir, "")
    model, processor, tokenizer, dtype, prompt_text = load_stage1_model(args)
    if model.temporal_model != args.method:
        raise ValueError(
            f"checkpoint temporal_model={model.temporal_model!r}, requested method={args.method!r}"
        )
    model.eval()
    dataset = HIVAUDataset(
        normalized_manifest,
        args.video_root,
        total_sampled_frames=FRAMES_PER_CLIP,
        sample_interval=SAMPLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        feature_cache_root=args.feature_cache_root,
        feature_cache_model_id=args.model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        require_spatial=(model.visual_fusion in SPATIAL_FUSION_MODES),
    )
    grouped = group_video_chunks(dataset.samples)
    with open(normalized_manifest) as handle:
        expected_video_ids = set(json.load(handle))
    if set(grouped) != expected_video_ids:
        missing = sorted(expected_video_ids - set(grouped))
        extra = sorted(set(grouped) - expected_video_ids)
        raise ValueError(f"dataset/cache coverage mismatch: missing={missing}, extra={extra}")

    save_json(manifest_path, {
        **protocol,
        "status": "incomplete",
        "num_videos": len(grouped),
        "videos": [],
    })

    entries = []
    for video_id, refs in tqdm(grouped.items(), desc=f"Extract {args.method} SCORE hidden"):
        cache_path = output_dir / f"{video_id}.npz"
        existing = _existing_video(cache_path, args.method, video_id) if args.resume else None
        if existing is not None:
            if existing["storage_dtype"] != args.storage_dtype:
                raise ValueError(
                    f"{cache_path}: stored dtype={existing['storage_dtype']}, "
                    f"requested {args.storage_dtype}; use --overwrite"
                )
            entries.append({
                "video_id": video_id,
                "file": cache_path.name,
                "num_windows": len(existing["window_index"]),
                "hidden_dim": existing["score_hidden"].shape[1],
            })
            continue
        with torch.inference_mode():
            result = infer_video(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                dataset=dataset,
                refs=refs,
                device=args.device,
                dtype=dtype,
                prompt_text=prompt_text,
                output_dir=output_dir,
                gt_root=args.gt_root,
                debug_state=False,
                collect_score_hidden=True,
                write_outputs=False,
            )
        rows = result["window_rows"]
        hidden = result["score_hidden"].numpy().astype(np.float32, copy=False)
        save_repr_video(
            cache_path,
            method=args.method,
            video_id=video_id,
            n_frames=result["n_frames"],
            fps=result["fps"],
            window_index=np.asarray([row["window_index"] for row in rows], dtype=np.int64),
            start_frame=np.asarray([row["start_frame"] for row in rows], dtype=np.int64),
            valid_end_frame=np.asarray([row["end_frame"] for row in rows], dtype=np.int64),
            score_hidden=hidden,
            storage_dtype=args.storage_dtype,
        )
        entries.append({
            "video_id": video_id,
            "file": cache_path.name,
            "num_windows": len(rows),
            "hidden_dim": hidden.shape[1],
        })

    entries.sort(key=lambda item: item["video_id"])
    manifest = {
        **protocol,
        "status": "complete",
        "num_videos": len(entries),
        "videos": entries,
    }
    save_json(manifest_path, manifest)
    print(f"Saved complete {args.method} representation cache: {output_dir}")


if __name__ == "__main__":
    main()
