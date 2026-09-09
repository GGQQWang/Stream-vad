"""Export blind human-evaluation text samples for LVLM inference variants."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hivau_dataset import HIVAUDataset, hivau_collate
from infer_stage1_ucf import (  # noqa: E402
    FRAMES_PER_CLIP,
    MAX_PIXELS,
    MAX_WINDOWS,
    MIN_PIXELS,
    SAMPLE_INTERVAL,
    _find_embed,
    load_gt,
    load_stage1_model,
    normalize_manifest,
)
from mil_utils import group_video_chunks  # noqa: E402
from tools.benchmark_lvlm_inference import (  # noqa: E402
    decode_generated,
    maybe_generate_texts,
    score_single_window,
)


METHODS = ("forward_semantic", "ar16", "ar64")
SAMPLING_PROTOCOL = (
    "GT-balanced fixed-seed window sampling; 100 abnormal and 100 normal by default; "
    "selected videos are replayed from their first window with strict single-window causal SSM."
)


@dataclass(frozen=True)
class WindowSample:
    sample_id: str
    video_id: str
    dataset_index: int
    local_window: int
    window_idx: int
    start_frame: int
    end_frame: int
    fps: float
    gt_label: int

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps


def _require_one_state(states: torch.Tensor, context: str) -> None:
    if states.ndim != 2 or states.shape[0] != 1:
        raise ValueError(f"{context} requires batch size 1, got shape={tuple(states.shape)}")


def build_forward_semantic_inputs(model, embed_fn, states: torch.Tensor) -> dict:
    """Build the no-teacher-forcing prefix matching summary CE training."""
    _require_one_state(states, "forward_semantic")
    llm_weight = embed_fn.weight
    device = llm_weight.device
    dtype = llm_weight.dtype
    states = states.to(device=device, dtype=dtype)
    query = model.summary_query.to(device=device, dtype=dtype).reshape(1, 1, -1)
    inputs_embeds = torch.cat([states.unsqueeze(1), query], dim=1)
    attention_mask = torch.ones(1, 2, dtype=torch.bool, device=device)
    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def generate_forward_semantic_text(model, embed_fn, tokenizer, states: torch.Tensor, max_new_tokens: int) -> str:
    _require_one_state(states, "forward_semantic")
    gen_inputs = build_forward_semantic_inputs(model, embed_fn, states)
    sequences = model.qwen.generate(
        **gen_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    texts = decode_generated(tokenizer, sequences, max_new_tokens)
    return texts[0] if texts else ""


def _window_label(gt, start_frame: int, end_frame: int) -> int:
    if end_frame <= start_frame:
        return 0
    return int(gt[start_frame:end_frame].max())


def collect_window_candidates(dataset: HIVAUDataset, grouped, gt_root: str | Path) -> tuple[List[WindowSample], List[WindowSample]]:
    normal: List[WindowSample] = []
    abnormal: List[WindowSample] = []
    for video_id, refs in grouped.items():
        first = dataset.samples[refs[0].index]
        n_frames = int(first["n_frames"])
        fps = float(first["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{video_id}: invalid fps={fps!r}")
        gt = load_gt(gt_root, video_id, n_frames)
        for ref in refs:
            meta = dataset.samples[ref.index]
            for offset, (start_frame, end_frame) in enumerate(
                zip(meta["window_start_frames"], meta["valid_end_frames"])
            ):
                window_idx = int(meta["chunk_start"]) + offset
                label = _window_label(gt, int(start_frame), min(int(end_frame), n_frames))
                sample = WindowSample(
                    sample_id="",
                    video_id=video_id,
                    dataset_index=ref.index,
                    local_window=offset,
                    window_idx=window_idx,
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    fps=fps,
                    gt_label=label,
                )
                if label == 1:
                    abnormal.append(sample)
                else:
                    normal.append(sample)
    return normal, abnormal


def sample_balanced_windows(
    normal: List[WindowSample],
    abnormal: List[WindowSample],
    *,
    per_class: int,
    seed: int,
) -> List[WindowSample]:
    if len(normal) < per_class or len(abnormal) < per_class:
        raise ValueError(
            f"insufficient GT-balanced windows: normal={len(normal)}, abnormal={len(abnormal)}, "
            f"required_each={per_class}"
        )
    rng = random.Random(seed)
    chosen = list(rng.sample(abnormal, per_class)) + list(rng.sample(normal, per_class))
    rng.shuffle(chosen)
    return [
        WindowSample(
            sample_id=f"sample_{idx:04d}",
            video_id=s.video_id,
            dataset_index=s.dataset_index,
            local_window=s.local_window,
            window_idx=s.window_idx,
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            fps=s.fps,
            gt_label=s.gt_label,
        )
        for idx, s in enumerate(chosen)
    ]


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_blind_outputs(raw_records: List[dict], seed: int) -> tuple[List[dict], dict]:
    rng = random.Random(seed)
    blind_records = []
    key = {"seed": seed, "samples": {}}
    for record in raw_records:
        labels = ["A", "B", "C"]
        methods = list(METHODS)
        rng.shuffle(methods)
        letter_to_method = dict(zip(labels, methods))
        key["samples"][record["sample_id"]] = letter_to_method
        blind_records.append({
            "sample_id": record["sample_id"],
            "video_id": record["video_id"],
            "window_idx": record["window_idx"],
            "start_frame": record["start_frame"],
            "end_frame": record["end_frame"],
            "start_sec": record["start_sec"],
            "end_sec": record["end_sec"],
            "gt_label": record["gt_label"],
            "score_prob": record["score_prob"],
            "texts": {letter: record[f"{method}_text"] for letter, method in letter_to_method.items()},
        })
    return blind_records, key


def export_records(
    *,
    model,
    tokenizer,
    dataset: HIVAUDataset,
    grouped,
    selected: List[WindowSample],
    device: torch.device,
    dtype: torch.dtype,
    prompt_text: str,
    forward_semantic_max_new_tokens: int,
) -> List[dict]:
    embed_fn = _find_embed(model.qwen)
    selected_by_video: Dict[str, Dict[int, WindowSample]] = {}
    for sample in selected:
        selected_by_video.setdefault(sample.video_id, {})[sample.window_idx] = sample
    selected_video_ids = set(selected_by_video)
    raw_by_sample_id: Dict[str, dict] = {}

    with torch.no_grad():
        for video_id, refs in tqdm(grouped.items(), desc="Export human-eval texts"):
            if video_id not in selected_video_ids:
                continue
            ssm_cache: dict = {}
            needed = selected_by_video[video_id]
            max_needed_window = max(needed)
            for ref in refs:
                batch = hivau_collate([dataset[ref.index]])
                if "features" not in batch:
                    raise RuntimeError("human-eval export requires cached visual features")
                valid_mask = batch["valid_mask"].to(device)
                valid_b, valid_w = valid_mask.nonzero(as_tuple=True)
                for b_t, w_t in zip(valid_b.cpu(), valid_w.cpu()):
                    b = int(b_t.item())
                    w = int(w_t.item())
                    if b != 0:
                        raise RuntimeError(f"{video_id}: expected batch size 1, got batch index {b}")
                    window_idx = int(batch["chunk_start"][b]) + w
                    if window_idx > max_needed_window:
                        break
                    single_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
                    single_window = batch["features"][:, w:w + 1].to(device=device, dtype=dtype)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                        state_emb, _, _, ssm_cache = model.encode_window_features(
                            single_window,
                            single_mask,
                            batch["video_id"],
                            ssm_cache,
                            training=False,
                        )
                        states = state_emb[:, 0]
                        logits = score_single_window(model, embed_fn, tokenizer, states, prompt_text)
                        score_prob = float(torch.sigmoid(logits).detach().float().cpu()[0].item())
                        if window_idx in needed:
                            sample = needed[window_idx]
                            forward_text = generate_forward_semantic_text(
                                model,
                                embed_fn,
                                tokenizer,
                                states,
                                forward_semantic_max_new_tokens,
                            )
                            ar16_texts = maybe_generate_texts(model, embed_fn, tokenizer, states, prompt_text, 16)
                            ar64_texts = maybe_generate_texts(model, embed_fn, tokenizer, states, prompt_text, 64)
                            raw_by_sample_id[sample.sample_id] = {
                                "sample_id": sample.sample_id,
                                "video_id": sample.video_id,
                                "window_idx": sample.window_idx,
                                "start_frame": sample.start_frame,
                                "end_frame": sample.end_frame,
                                "start_sec": sample.start_sec,
                                "end_sec": sample.end_sec,
                                "gt_label": sample.gt_label,
                                "score_prob": score_prob,
                                "forward_semantic_text": forward_text,
                                "ar16_text": ar16_texts[0] if ar16_texts else "",
                                "ar64_text": ar64_texts[0] if ar64_texts else "",
                            }
                if int(batch["chunk_start"][0]) > max_needed_window:
                    break
            ssm_cache.pop(video_id, None)

    missing = [s.sample_id for s in selected if s.sample_id not in raw_by_sample_id]
    if missing:
        raise RuntimeError(f"failed to export selected samples: {missing[:10]}")
    return [raw_by_sample_id[s.sample_id] for s in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", default="")
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", default="/data3/wgq/outputs/lvlm_human_eval_texts")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--forward-semantic-max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    args.device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_manifest = normalize_manifest(args.test_manifest, output_dir, args.video_id)
    model, _, tokenizer, dtype, prompt_text = load_stage1_model(args)
    model.eval()
    if not hasattr(model, "summary_query"):
        raise RuntimeError("checkpoint/model does not expose summary_query; cannot export forward_semantic")

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
    )
    grouped = group_video_chunks(dataset.samples)
    if args.video_id:
        selected_video_id = Path(args.video_id).stem
        if selected_video_id not in grouped:
            raise ValueError(f"video_id={selected_video_id!r} not found in dataset")
        grouped = {selected_video_id: grouped[selected_video_id]}

    normal, abnormal = collect_window_candidates(dataset, grouped, args.gt_root)
    selected = sample_balanced_windows(
        normal,
        abnormal,
        per_class=args.samples_per_class,
        seed=args.seed,
    )
    raw_records = export_records(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        grouped=grouped,
        selected=selected,
        device=args.device,
        dtype=dtype,
        prompt_text=prompt_text,
        forward_semantic_max_new_tokens=args.forward_semantic_max_new_tokens,
    )

    with open(output_dir / "human_eval_raw.jsonl", "w") as f:
        for record in raw_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    blind_records, key = build_blind_outputs(raw_records, args.seed)
    write_json(output_dir / "human_eval_blind.json", {"samples": blind_records})
    write_json(output_dir / "human_eval_key.json", key)
    write_json(output_dir / "human_eval_manifest.json", {
        "seed": args.seed,
        "abnormal_count": args.samples_per_class,
        "normal_count": args.samples_per_class,
        "checkpoint": args.stage1_dir,
        "model_path": args.model_path,
        "sampling_protocol": SAMPLING_PROTOCOL,
        "forward_semantic_prefix": "[state, summary_query]",
        "ar_prefix": "[state, prompt]",
        "forward_semantic_max_new_tokens": args.forward_semantic_max_new_tokens,
        "methods": list(METHODS),
    })


if __name__ == "__main__":
    main()
