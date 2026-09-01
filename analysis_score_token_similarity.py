"""Compare Full vs No-SSM score-token hidden representations around ALL
GT anomaly boundaries (0->1 onsets and 1->0 offsets) for ONE video.

For every window in true temporal order:

  full_t   = x_t + alpha * Adapter(SSM_out_t)
             -> Qwen -> last-layer score_query hidden
             (``out.hidden_states[-1][:, -1, :]``, pre-score_head)
  no_ssm_t = x_t
             -> the SAME Qwen / LoRA / prompt / score_query
             -> the same score_query hidden

The only difference between the two runs is the SSM residual term.  The
causal SSM state is carried across chunks exactly as in inference (one
ssm_cache per video, cleared at the end), and the SSM is run only ONCE
per chunk: both representations come from the same
``encode_window_features`` call (Full = its first return value,
No-SSM = its second, the pre-residual base).

The prompt text is read from the checkpoint itself (``load_stage1_model``
returns it), so it is identical to the one used at inference time.

Run:
    python -u analysis_score_token_similarity.py \
      --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
      --stage1-dir /path/to/checkpoint \
      --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
      --video-root /data1/wjq/data/UCF_Crime/testing/videos \
      --feature-cache-root /data3/wgq/data/ucf_original_all_visual_cache_100352 \
      --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
      --video-id <VIDEO_ID> \
      --boundary-k 5 \
      --output-dir /data3/wgq/outputs/score_similarity/<VIDEO_ID> \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hivau_dataset import HIVAUDataset, hivau_collate
from infer_stage1_ucf import load_gt, load_stage1_model, normalize_manifest
from mil_utils import group_video_chunks

# same windowing as infer_stage1_ucf.py (validated against the checkpoint)
FRAMES_PER_CLIP = 16
SAMPLE_INTERVAL = 3
MAX_WINDOWS = 8
MIN_PIXELS = 100352
MAX_PIXELS = 100352


def _pairwise_stats(S: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray,
                    exclude_diagonal: bool) -> Tuple[float, float, float]:
    """Mean cosine over intra-a, intra-b and cross a-b pairs.

    ``exclude_diagonal`` only affects the intra blocks (i<j upper
    triangle) so self-similarity 1.0 does not inflate NN/AA.
    """
    if exclude_diagonal:
        if len(idx_a) >= 2:
            tri = np.triu_indices(len(idx_a), k=1)
            aa = S[np.ix_(idx_a, idx_a)][tri]
        else:
            aa = np.empty(0)
        if len(idx_b) >= 2:
            tri = np.triu_indices(len(idx_b), k=1)
            bb = S[np.ix_(idx_b, idx_b)][tri]
        else:
            bb = np.empty(0)
    else:
        aa = S[np.ix_(idx_a, idx_a)].ravel() if len(idx_a) else np.empty(0)
        bb = S[np.ix_(idx_b, idx_b)].ravel() if len(idx_b) else np.empty(0)
    cross = S[np.ix_(idx_a, idx_b)].ravel() if (len(idx_a) and len(idx_b)) else np.empty(0)
    return (
        float(aa.mean()) if len(aa) else float("nan"),
        float(bb.mean()) if len(bb) else float("nan"),
        float(cross.mean()) if len(cross) else float("nan"),
    )


def _classify_window_gt(window_gt: np.ndarray) -> str:
    """Content-only window classification (no onset argument).

    all-0 -> pure_normal, all-1 -> pure_abnormal, mixed -> transition,
    empty -> other (defensive; never vacuous-truth pure_abnormal).
    """
    if window_gt.size == 0:
        return "other"
    if np.all(window_gt == 0):
        return "pure_normal"
    if np.all(window_gt == 1):
        return "pure_abnormal"
    return "transition"


def _split_indices(win_rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Window indices for the global separation metrics, by gt_state.

    transition and other windows are excluded from both sets.
    """
    normal_idx = np.array(
        [i for i, r in enumerate(win_rows) if r["gt_state"] == "pure_normal"],
        dtype=np.int64,
    )
    abnormal_idx = np.array(
        [i for i, r in enumerate(win_rows) if r["gt_state"] == "pure_abnormal"],
        dtype=np.int64,
    )
    return normal_idx, abnormal_idx


def _score_token_hidden(model, embed_fn, tokenizer, prompt_text: str,
                        states: torch.Tensor) -> torch.Tensor:
    """Replicate ``forward_score_token``'s input construction, but return
    the last-layer score_query hidden instead of the scalar score.

    The score representation is exactly
    ``out.hidden_states[-1][:, -1, :]`` (pre-score_head): no scalar
    score, no mean-pool, no lm_head logits.
    """
    N = states.shape[0]
    if N == 0:
        return states.new_zeros(0, states.shape[-1])

    llm_weight = embed_fn.weight
    llm_device = llm_weight.device
    llm_dtype = llm_weight.dtype
    states = states.to(device=llm_device, dtype=llm_dtype)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=llm_device)
    prompt_emb = embed_fn(prompt_ids_t).unsqueeze(0).expand(N, -1, -1)
    query = model.score_query.to(device=llm_device, dtype=llm_dtype).reshape(1, 1, -1).expand(N, 1, -1)
    inputs = torch.cat([states.unsqueeze(1), prompt_emb, query], dim=1)
    attn = torch.ones(N, inputs.shape[1], dtype=torch.bool, device=llm_device)
    out = model.qwen(
        inputs_embeds=inputs,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[-1][:, -1, :]          # [N, H] pre-score_head


def _find_boundaries(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ALL 0->1 onsets and 1->0 offsets (first frame of the new state)."""
    d = np.diff(gt.astype(np.int64))
    onsets = np.where(d == 1)[0] + 1
    offsets = np.where(d == -1)[0] + 1
    return onsets, offsets


def _boundary_heatmap_pos(win_rows: List[dict], frame: int) -> Optional[float]:
    """Map a GT boundary frame into heatmap cell coordinates.

    The line lands at ``row_idx - 0.5 + frac`` where frac is the frame's
    proportional position INSIDE its covering window (a boundary inside
    a window is not pushed to the window edge).  Returns None when no
    window covers the frame.
    """
    for i, r in enumerate(win_rows):
        if r["start_frame"] <= frame < r["valid_end_frame"]:
            frac = (frame - r["start_frame"]) / (r["valid_end_frame"] - r["start_frame"])
            return i - 0.5 + frac
    return None


def _boundary_window_idx(win_rows: List[dict], frame: int, kind: str,
                         K: int, left_limit: int,
                         right_limit: int) -> Tuple[List[int], List[int]]:
    """K nearest pure windows on each side of ONE GT boundary, restricted
    to the interval between its ADJACENT GT boundaries.

    left_limit  = previous boundary frame (0 = video start if none)
    right_limit = next boundary frame (video end if none)

    onset  (0->1): left = pure_normal, right = pure_abnormal
    offset (1->0): left = pure_abnormal, right = pure_normal

    left  windows must lie in [left_limit, frame]
    right windows must lie in [frame, right_limit]
    so a boundary's local statistics can never cross the neighbouring
    anomaly segments.  transition and other windows are always skipped.
    """
    left_state, right_state = (
        ("pure_normal", "pure_abnormal") if kind == "onset"
        else ("pure_abnormal", "pure_normal")
    )
    left = [i for i, r in enumerate(win_rows)
            if r["gt_state"] == left_state
            and r["start_frame"] >= left_limit
            and r["valid_end_frame"] <= frame]
    right = [i for i, r in enumerate(win_rows)
             if r["gt_state"] == right_state
             and r["start_frame"] >= frame
             and r["valid_end_frame"] <= right_limit]
    return left[-K:], right[:K]


def _boundary_gap(S: np.ndarray, win_rows: List[dict], frame: int,
                  kind: str, K: int, left_limit: int, right_limit: int,
                  video_id: str) -> float:
    """Local gap around ONE GT boundary, confined to its adjacent
    boundaries: (within_left + within_right) / 2 - cross.

    Returns NaN (with a warning) when either side has fewer than 2 pure
    windows INSIDE the interval; windows are never borrowed across the
    neighbouring boundaries.
    """
    left, right = _boundary_window_idx(
        win_rows, frame, kind, K, left_limit, right_limit,
    )
    if len(left) < 2 or len(right) < 2:
        print(
            f"WARNING {video_id}: {kind}@{frame} boundary_gap needs >=2 "
            f"windows on each side within [{left_limit}, {right_limit}] "
            f"(left={len(left)}, right={len(right)}); returning NaN"
        )
        return float("nan")
    within_left, _, _ = _pairwise_stats(S, np.array(left), np.array(left),
                                        exclude_diagonal=True)
    within_right, _, _ = _pairwise_stats(S, np.array(right), np.array(right),
                                         exclude_diagonal=True)
    _, _, cross = _pairwise_stats(S, np.array(left), np.array(right),
                                  exclude_diagonal=False)
    if np.isnan(within_left) or np.isnan(within_right) or np.isnan(cross):
        return float("nan")
    return (within_left + within_right) / 2.0 - cross


def _run_logic_self_check() -> None:
    """Cheap CPU-only checks of the classification / boundary logic."""
    # content-only classification
    z48 = np.zeros(48, dtype=np.int64)
    o48 = np.ones(48, dtype=np.int64)
    m48 = z48.copy()
    m48[:18] = 1
    assert _classify_window_gt(z48) == "pure_normal"
    assert _classify_window_gt(o48) == "pure_abnormal"
    assert _classify_window_gt(m48) == "transition"
    assert _classify_window_gt(np.zeros(0, dtype=np.int64)) == "other"

    rows = [
        {"window_index": 0, "start_frame": 0, "valid_end_frame": 48,
         "binary_gt": 0, "gt_state": "pure_normal"},
        {"window_index": 1, "start_frame": 48, "valid_end_frame": 96,
         "binary_gt": 1, "gt_state": "transition"},
        {"window_index": 2, "start_frame": 96, "valid_end_frame": 144,
         "binary_gt": 1, "gt_state": "pure_abnormal"},
        {"window_index": 3, "start_frame": 144, "valid_end_frame": 192,
         "binary_gt": 1, "gt_state": "pure_abnormal"},
        {"window_index": 4, "start_frame": 192, "valid_end_frame": 240,
         "binary_gt": 0, "gt_state": "other"},
        {"window_index": 5, "start_frame": 240, "valid_end_frame": 288,
         "binary_gt": 0, "gt_state": "pure_normal"},
    ]
    # split: transition/other excluded from both sets
    normal_idx, abnormal_idx = _split_indices(rows)
    assert 1 not in normal_idx and 4 not in normal_idx
    assert 1 not in abnormal_idx and 4 not in abnormal_idx
    # onset@48 within [0, 288]: left = nearest pure_normal, right = nearest
    # pure_abnormal (transition/other skipped)
    left, right = _boundary_window_idx(rows, 48, "onset", 2, 0, 288)
    assert left == [0] and right == [2, 3]
    # offset@192 within [48, 288]: sides swapped
    left, right = _boundary_window_idx(rows, 192, "offset", 2, 48, 288)
    assert left == [2, 3] and right == [5]
    # fractional heatmap mapping: frame 210 inside [192,240) at row 4
    pos = _boundary_heatmap_pos(rows, 210)
    assert pos is not None and abs(pos - (4 - 0.5 + 18 / 48)) < 1e-9
    # frame exactly on a window edge maps to the cell boundary (row-0.5)
    pos_edge = _boundary_heatmap_pos(rows, 96)
    assert pos_edge is not None and abs(pos_edge - (2 - 0.5)) < 1e-9

    # multi-boundary video: N -> onset1@48 -> A -> offset1@192 -> N ->
    # onset2@288 -> A.  Each boundary's windows must stay inside its own
    # adjacent-boundary interval and never reach the other segment.
    multi = [
        {"window_index": i, "start_frame": i * 48,
         "valid_end_frame": (i + 1) * 48, "binary_gt": 1, "gt_state": s}
        for i, s in enumerate([
            "pure_normal",           # [0,48)
            "transition",            # [48,96)    contains onset1@48
            "pure_abnormal",         # [96,144)
            "pure_abnormal",         # [144,192)
            "transition",            # [192,240)  contains offset1@192
            "pure_normal",           # [240,288)
            "transition",            # [288,336)  contains onset2@288
            "pure_abnormal",         # [336,384)
            "pure_abnormal",         # [384,432)
        ])
    ]
    # onset1@48 (limits 0..192): right may NOT reach onset2's abnormals
    left, right = _boundary_window_idx(multi, 48, "onset", 2, 0, 192)
    assert left == [0] and right == [2, 3]
    assert 7 not in right and 8 not in right
    # offset1@192 (limits 48..288): both sides confined to segment 1
    left, right = _boundary_window_idx(multi, 192, "offset", 2, 48, 288)
    assert left == [2, 3] and right == [5]
    assert 7 not in left and 8 not in left and 7 not in right and 8 not in right
    # onset2@288 (limits 192..432): left may NOT reach onset1's normals
    left, right = _boundary_window_idx(multi, 288, "onset", 2, 192, 432)
    assert left == [5] and right == [7, 8]
    assert 0 not in left
    print("logic self-check OK (12 cases)")


def _draw_heatmap(S: np.ndarray, boundaries: List[Tuple[float, str]],
                  title: str, out_path: Path, vmin: float,
                  vmax: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = S.shape[0]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(S, cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    # ALL GT boundaries: onset (0->1) red, offset (1->0) blue
    for pos, kind in boundaries:
        color = "red" if kind == "onset" else "blue"
        ax.axvline(pos, color=color, linewidth=2, linestyle="--")
        ax.axhline(pos, color=color, linewidth=2, linestyle="--")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Window index")
    ax.set_title(title, fontsize=12)
    # tick density for long videos
    step = max(1, T // 20)
    ax.set_xticks(np.arange(0, T, step))
    ax.set_yticks(np.arange(0, T, step))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--feature-cache-root", default="")
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_logic_self_check()

    # ---- normalize manifest (reuse the infer script's logic) ----
    normalized_manifest = normalize_manifest(
        args.test_manifest, output_dir, args.video_id,
    )

    # ---- load the stage1 checkpoint exactly like inference (prompt_text
    # comes from the checkpoint -> identical to the inference prompt) ----
    model, processor, tokenizer, dtype, prompt_text = load_stage1_model(args)
    model.eval()

    # ---- dataset + per-video chunks in temporal order ----
    dataset = HIVAUDataset(
        normalized_manifest,
        args.video_root,
        total_sampled_frames=FRAMES_PER_CLIP,
        sample_interval=SAMPLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        feature_cache_root=args.feature_cache_root or None,
        feature_cache_model_id=args.model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    grouped = group_video_chunks(dataset.samples)
    video_id = Path(args.video_id).stem
    if video_id not in grouped:
        raise ValueError(f"video_id={video_id!r} not found in dataset")
    refs = grouped[video_id]                      # already sorted by chunk_start

    n_frames = int(dataset.samples[refs[0].index]["n_frames"])
    gt = load_gt(args.gt_root, video_id, n_frames)

    # ---- ALL GT boundaries: 0->1 onsets and 1->0 offsets ----
    onsets, offsets = _find_boundaries(gt)
    if len(onsets) == 0 and len(offsets) == 0:
        print(
            f"video_id={video_id}: no 0->1 / 1->0 GT boundary found; "
            "skipping (nothing to analyse)"
        )
        return

    embed_fn = model.qwen.get_input_embeddings()

    # ---- per-window extraction with continuous causal SSM state ----
    full_list: List[np.ndarray] = []
    no_list: List[np.ndarray] = []
    win_rows: List[dict] = []
    ssm_cache: dict = {}

    with torch.no_grad():
        for ref in tqdm(refs, desc=f"Extract {video_id}", leave=False):
            batch = hivau_collate([dataset[ref.index]])
            valid_mask_cpu = batch["valid_mask"]
            valid_mask = valid_mask_cpu.to(args.device)
            window_batch = batch["features"].to(
                device=args.device, dtype=dtype,
            )
            valid_b_cpu, valid_w_cpu = valid_mask_cpu.nonzero(as_tuple=True)
            with torch.autocast(device_type=args.device.type, dtype=dtype,
                                enabled=(args.device.type == "cuda")):
                # ONE causal SSM pass per chunk: Full = first return,
                # No-SSM base = second return (same x_t)
                state_emb, window_base, _, ssm_cache = (
                    model.encode_window_features(
                        window_batch, valid_mask, batch["video_id"],
                        ssm_cache, training=False,
                    )
                )
                if len(valid_b_cpu) == 0:
                    continue
                valid_b, valid_w = valid_mask.nonzero(as_tuple=True)
                h_full = _score_token_hidden(
                    model, embed_fn, tokenizer, prompt_text,
                    state_emb[valid_b, valid_w],
                )
                h_no = _score_token_hidden(
                    model, embed_fn, tokenizer, prompt_text,
                    window_base[valid_b, valid_w],
                )
            start_frames = batch["window_start_frames"]
            valid_end_frames = batch["valid_end_frames"]
            for i in range(len(valid_b_cpu)):
                w = int(valid_w_cpu[i].item())
                global_idx = int(batch["chunk_start"][0]) + w
                start = int(start_frames[0, w].item())
                end = int(valid_end_frames[0, w].item())
                window_gt = gt[start:end]
                bin_gt = int(window_gt.any())
                full_list.append(h_full[i].detach().float().cpu().numpy())
                no_list.append(h_no[i].detach().float().cpu().numpy())
                win_rows.append({
                    "window_index": global_idx,
                    "start_frame": start,
                    "valid_end_frame": end,
                    "binary_gt": bin_gt,
                    "gt_state": _classify_window_gt(window_gt),
                })
        ssm_cache.clear()

    # ---- enforce true temporal order ----
    order = sorted(range(len(win_rows)), key=lambda i: win_rows[i]["window_index"])
    win_rows = [win_rows[i] for i in order]
    F_full = np.stack([full_list[i] for i in order], axis=0)    # [T, H]
    F_no = np.stack([no_list[i] for i in order], axis=0)        # [T, H]
    T = len(win_rows)
    if T == 0:
        raise RuntimeError(f"{video_id}: no valid windows extracted")

    # ---- cosine similarity matrices ----
    Fn_full = F.normalize(torch.from_numpy(F_full), dim=-1).numpy()
    Fn_no = F.normalize(torch.from_numpy(F_no), dim=-1).numpy()
    S_full = Fn_full @ Fn_full.T
    S_no = Fn_no @ Fn_no.T

    # ---- global NN / AA / NA (transition and other fully excluded) ----
    normal_idx, abnormal_idx = _split_indices(win_rows)
    num_pure_normal = int((np.array([r["gt_state"] for r in win_rows]) == "pure_normal").sum())
    num_transition = int((np.array([r["gt_state"] for r in win_rows]) == "transition").sum())
    num_pure_abnormal = int((np.array([r["gt_state"] for r in win_rows]) == "pure_abnormal").sum())
    num_other = int((np.array([r["gt_state"] for r in win_rows]) == "other").sum())
    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        raise RuntimeError(
            f"{video_id}: need both pure_normal and pure_abnormal windows "
            f"(normal={len(normal_idx)}, abnormal={len(abnormal_idx)})"
        )

    def _sep(S):
        nn, aa, na = _pairwise_stats(S, normal_idx, abnormal_idx,
                                     exclude_diagonal=True)
        # _pairwise_stats computes (intra-a=normal, intra-b=abnormal, cross)
        return nn, aa, na, (nn + aa) / 2.0 - na

    full_nn, full_aa, full_na, full_delta = _sep(S_full)
    no_nn, no_aa, no_na, no_delta = _sep(S_no)
    delta_sep_gain = full_delta - no_delta

    # ---- per-boundary local gap (ALL onsets + offsets) ----
    boundaries = ([(int(f), "onset") for f in onsets]
                  + [(int(f), "offset") for f in offsets])
    boundaries.sort(key=lambda b: b[0])
    boundary_specs: List[dict] = []
    for frame, kind in boundaries:
        pos = _boundary_heatmap_pos(win_rows, frame)
        if pos is None:
            print(
                f"WARNING {video_id}: {kind} boundary frame {frame} is not "
                "covered by any window; excluded from heatmap"
            )
        boundary_specs.append({"frame": frame, "kind": kind, "pos": pos})

    boundary_rows: List[dict] = []
    gaps_full: List[float] = []
    gaps_no: List[float] = []
    for bi, spec in enumerate(boundary_specs):
        frame, kind = spec["frame"], spec["kind"]
        # local interval: between the ADJACENT GT boundaries only, so a
        # boundary's gap can never borrow windows from another segment
        left_limit = boundaries[bi - 1][0] if bi > 0 else 0
        right_limit = boundaries[bi + 1][0] if bi < len(boundaries) - 1 else n_frames
        gap_full = _boundary_gap(S_full, win_rows, frame, kind,
                                 args.boundary_k, left_limit, right_limit,
                                 video_id)
        gap_no = _boundary_gap(S_no, win_rows, frame, kind,
                               args.boundary_k, left_limit, right_limit,
                               video_id)
        left, right = _boundary_window_idx(
            win_rows, frame, kind, args.boundary_k, left_limit, right_limit,
        )
        both_valid = not (np.isnan(gap_full) or np.isnan(gap_no))
        if not np.isnan(gap_full):
            gaps_full.append(gap_full)
        if not np.isnan(gap_no):
            gaps_no.append(gap_no)
        boundary_rows.append({
            "boundary_index": bi,
            "boundary_type": kind,
            "boundary_frame": frame,
            "heatmap_pos": (round(spec["pos"], 6) if spec["pos"] is not None else None),
            "num_left": len(left),
            "num_right": len(right),
            "gap_full": gap_full,
            "gap_no_ssm": gap_no,
            "gap_gain": (gap_full - gap_no) if both_valid else float("nan"),
        })
    mean_gap_full = float(np.mean(gaps_full)) if gaps_full else float("nan")
    mean_gap_no = float(np.mean(gaps_no)) if gaps_no else float("nan")
    mean_gap_gain = mean_gap_full - mean_gap_no
    if not gaps_full:
        print(
            f"WARNING {video_id}: no boundary produced a valid gap; "
            "mean_boundary_gap_* = NaN"
        )

    metrics = {
        "video_id": video_id,
        "num_windows": T,
        "num_pure_normal_windows": num_pure_normal,
        "num_transition_windows": num_transition,
        "num_pure_abnormal_windows": num_pure_abnormal,
        "num_other_windows": num_other,
        "num_onset_boundaries": len(onsets),
        "num_offset_boundaries": len(offsets),
        "full_score_hidden": {
            "NN": full_nn,
            "AA": full_aa,
            "NA": full_na,
            "delta_sep": full_delta,
        },
        "no_ssm_score_hidden": {
            "NN": no_nn,
            "AA": no_aa,
            "NA": no_na,
            "delta_sep": no_delta,
        },
        "delta_sep_gain": delta_sep_gain,
        "mean_boundary_gap_full": mean_gap_full,
        "mean_boundary_gap_no_ssm": mean_gap_no,
        "mean_boundary_gap_gain": mean_gap_gain,
    }

    # ---- heatmaps (shared vmin/vmax; ALL boundaries drawn) ----
    common_min = float(min(S_full.min(), S_no.min()))
    common_max = float(max(S_full.max(), S_no.max()))
    heatmap_lines = [(spec["pos"], spec["kind"]) for spec in boundary_specs
                     if spec["pos"] is not None]
    n_onset_drawn = sum(1 for _, k in heatmap_lines if k == "onset")
    n_offset_drawn = sum(1 for _, k in heatmap_lines if k == "offset")
    legend = (f"GT boundaries: onset(0->1) red={n_onset_drawn}, "
              f"offset(1->0) blue={n_offset_drawn}")
    _draw_heatmap(
        S_full, heatmap_lines,
        f"{video_id} — Full x_t + alpha*Adapter(SSM_out) score-token "
        f"hidden cosine ({legend})",
        output_dir / "full_score_hidden_cosine_heatmap.png", common_min, common_max,
    )
    _draw_heatmap(
        S_no, heatmap_lines,
        f"{video_id} — No-SSM x_t score-token hidden cosine ({legend})",
        output_dir / "no_ssm_score_hidden_cosine_heatmap.png", common_min, common_max,
    )

    # ---- outputs ----
    np.save(output_dir / "full_score_hidden.npy", F_full)
    np.save(output_dir / "no_ssm_score_hidden.npy", F_no)
    np.save(output_dir / "full_score_hidden_cosine.npy", S_full)
    np.save(output_dir / "no_ssm_score_hidden_cosine.npy", S_no)

    with open(output_dir / "window_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["window_index", "start_frame",
                           "valid_end_frame", "binary_gt", "gt_state"],
        )
        writer.writeheader()
        for r in win_rows:
            writer.writerow(r)

    with open(output_dir / "boundary_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["boundary_index", "boundary_type",
                           "boundary_frame", "heatmap_pos",
                           "num_left", "num_right",
                           "gap_full", "gap_no_ssm", "gap_gain"],
        )
        writer.writeheader()
        for r in boundary_rows:
            writer.writerow(r)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- diagnostics ----
    print(
        f"video_id={video_id} num_windows={T} "
        f"pure_normal={num_pure_normal} transition={num_transition} "
        f"pure_abnormal={num_pure_abnormal} other={num_other}"
    )
    print(f"boundaries: onsets={len(onsets)} offsets={len(offsets)}")
    print(
        f"full:   NN={full_nn:.4f} AA={full_aa:.4f} NA={full_na:.4f} "
        f"delta_sep={full_delta:.4f}"
    )
    print(
        f"no_ssm: NN={no_nn:.4f} AA={no_aa:.4f} NA={no_na:.4f} "
        f"delta_sep={no_delta:.4f}"
    )
    print(f"delta_sep_gain={delta_sep_gain:.4f}")
    print(
        f"mean_boundary_gap_full={mean_gap_full} "
        f"mean_boundary_gap_no_ssm={mean_gap_no} "
        f"mean_boundary_gap_gain={mean_gap_gain}"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
