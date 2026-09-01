"""Analyze whether the SSM reorganizes normal/abnormal temporal
representations into a clearer separation around the anomaly onset.

For ONE video, extract per-window (in true temporal order):

  1. x_t          — the window-level visual feature BEFORE the SSM
                    (the mean-pooled compressed feature that enters
                    SSMBlock.forward_chunk, i.e. ``window_batch``
                    returned by ``encode_window_features``)
  2. h_internal_t — the SSM internal temporal representation
                    (out_norm output, pre-out_proj), obtained via
                    ``encode_window_features(..., return_internal=True)``

then compute pairwise cosine similarity matrices for both, draw two
heatmaps with the GT abnormal-onset boundary, and quantify the
normal/abnormal separation (global NN/AA/NA + local boundary gap).

The recurrent SSM state is carried across chunks exactly as in causal
inference (one ssm_cache per video, cleared at the end), so
h_internal_t is identical to the online streaming representation.

Run:
    python -u analysis_ssm_similarity.py \
      --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
      --stage1-dir /path/to/checkpoint \
      --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
      --video-root /data1/wjq/data/UCF_Crime/testing/videos \
      --feature-cache-root /data3/wgq/data/ucf_original_all_visual_cache_100352 \
      --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
      --video-id <VIDEO_ID> \
      --boundary-k 5 \
      --output-dir /data3/wgq/outputs/ssm_similarity/<VIDEO_ID> \
      --device cuda:0
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _classify_gt_state(start: int, end: int, onset: int, bin_gt: int) -> str:
    """Classify a window by its GT coverage relative to the anomaly onset.

    Edge cases: ``end == onset`` -> pure_normal; ``start == onset`` ->
    pure_abnormal (or other when the window has no abnormal frame); an
    onset exactly on a window boundary never creates a transition
    window.
    """
    if end <= onset:
        return "pure_normal"
    if start < onset < end:
        return "transition"
    if start >= onset and bin_gt:
        return "pure_abnormal"
    return "other"


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


def _boundary_idx(win_rows: List[dict], K: int) -> Tuple[List[int], List[int]]:
    """Indices for the local boundary gap.

    left  = the LAST K pure_normal windows (closest before the onset)
    right = the FIRST K pure_abnormal windows (earliest at/after onset)
    transition and other windows are never included.
    """
    left = [i for i, r in enumerate(win_rows) if r["gt_state"] == "pure_normal"][-K:]
    right = [i for i, r in enumerate(win_rows) if r["gt_state"] == "pure_abnormal"][:K]
    return left, right


def _run_logic_self_check() -> None:
    """Cheap CPU-only checks of the gt_state / boundary selection logic."""
    # Case 1: [0,48) with onset=48 -> pure_normal
    assert _classify_gt_state(0, 48, 48, 0) == "pure_normal"
    # Case 2: [48,96) with onset=48, GT abnormal -> pure_abnormal
    assert _classify_gt_state(48, 96, 48, 1) == "pure_abnormal"
    # Case 3: [32,80) with onset=48 -> transition
    assert _classify_gt_state(32, 80, 48, 1) == "transition"
    # Case 4: onset 之后的正常 window -> other
    assert _classify_gt_state(96, 144, 48, 0) == "other"

    # Case 5: transition/other never enter the global normal/abnormal sets
    rows = [
        {"window_index": 0, "gt_state": "pure_normal"},
        {"window_index": 1, "gt_state": "pure_normal"},
        {"window_index": 2, "gt_state": "pure_normal"},
        {"window_index": 3, "gt_state": "transition"},
        {"window_index": 4, "gt_state": "pure_abnormal"},
        {"window_index": 5, "gt_state": "pure_abnormal"},
        {"window_index": 6, "gt_state": "other"},
        {"window_index": 7, "gt_state": "pure_abnormal"},
    ]
    normal_idx, abnormal_idx = _split_indices(rows)
    assert 3 not in normal_idx and 6 not in normal_idx
    assert 3 not in abnormal_idx and 6 not in abnormal_idx
    # Case 6: boundary selection = last K pure_normal + first K pure_abnormal
    left, right = _boundary_idx(rows, K=2)
    assert left == [1, 2] and right == [4, 5]
    print("logic self-check OK (6 cases)")


def _boundary_stats(S: np.ndarray, win_rows: List[dict], K: int,
                    video_id: str) -> float:
    """Local boundary gap over gt_state-selected windows.

    left  = the LAST K pure_normal windows (before the onset)
    right = the FIRST K pure_abnormal windows (at/after the onset)
    transition and other windows are never included.
    """
    left, right = _boundary_idx(win_rows, K)
    if len(left) < 2 or len(right) < 2:
        print(
            f"WARNING {video_id}: boundary_gap needs >=2 windows on each "
            f"side (left={len(left)}, right={len(right)}); returning NaN"
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


def _draw_heatmap(S: np.ndarray, onset_window: int, title: str,
                  out_path: Path, vmin: float, vmax: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = S.shape[0]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(S, cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    # GT abnormal-onset boundary (between cells)
    ax.axvline(onset_window - 0.5, color="red", linewidth=2, linestyle="--")
    ax.axhline(onset_window - 0.5, color="red", linewidth=2, linestyle="--")
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

    # ---- load the stage1 checkpoint exactly like inference ----
    model, processor, tokenizer, dtype, _ = load_stage1_model(args)
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

    # ---- GT abnormal onset frame (first 0->1 transition) ----
    transition = np.where(np.diff(gt.astype(np.int64)) == 1)[0]
    if len(transition) == 0:
        print(
            f"video_id={video_id}: no 0->1 GT transition found; "
            "skipping (no anomaly onset to analyse)"
        )
        return
    anomaly_onset_frame = int(transition[0] + 1)   # the first abnormal frame

    # ---- per-window extraction with continuous causal SSM state ----
    x_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
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
            with torch.autocast(device_type=args.device.type, dtype=dtype,
                                enabled=(args.device.type == "cuda")):
                _, visual_windows, _, h_internal, ssm_cache = (
                    model.encode_window_features(
                        window_batch, valid_mask, batch["video_id"],
                        ssm_cache, training=False, return_internal=True,
                    )
                )
            start_frames = batch["window_start_frames"]
            valid_end_frames = batch["valid_end_frames"]
            valid_b, valid_w = valid_mask_cpu.nonzero(as_tuple=True)
            for i in range(len(valid_b)):
                w = int(valid_w[i].item())
                global_idx = int(batch["chunk_start"][0]) + w
                start = int(start_frames[0, w].item())
                end = int(valid_end_frames[0, w].item())
                bin_gt = int(gt[start:end].any())
                x_list.append(visual_windows[0, w].detach().float().cpu().numpy())
                h_list.append(h_internal[0, w].detach().float().cpu().numpy())
                win_rows.append({
                    "window_index": global_idx,
                    "start_frame": start,
                    "valid_end_frame": end,
                    "binary_gt": bin_gt,
                    "gt_state": _classify_gt_state(
                        start, end, anomaly_onset_frame, bin_gt,
                    ),
                })
        ssm_cache.clear()

    # ---- enforce true temporal order ----
    order = sorted(range(len(win_rows)), key=lambda i: win_rows[i]["window_index"])
    win_rows = [win_rows[i] for i in order]
    X = np.stack([x_list[i] for i in order], axis=0)     # [T, 3584]
    H = np.stack([h_list[i] for i in order], axis=0)     # [T, 256]
    T = len(win_rows)
    if T == 0:
        raise RuntimeError(f"{video_id}: no valid windows extracted")

    # ---- map onset frame to window: first window whose valid coverage
    # contains the onset frame ----
    onset_window = None
    for row in win_rows:
        if row["start_frame"] <= anomaly_onset_frame < row["valid_end_frame"]:
            onset_window = row["window_index"]
            break
    if onset_window is None:
        raise RuntimeError(
            f"{video_id}: anomaly onset frame {anomaly_onset_frame} is not "
            "covered by any window"
        )
    onset_pos = win_rows.index(
        next(r for r in win_rows if r["window_index"] == onset_window)
    )

    # ---- cosine similarity matrices ----
    Xn = F.normalize(torch.from_numpy(X), dim=-1).numpy()
    Hn = F.normalize(torch.from_numpy(H), dim=-1).numpy()
    S_pre = Xn @ Xn.T
    S_ssm = Hn @ Hn.T

    # ---- normal / abnormal window split (gt_state based; transition and
    # other windows are fully excluded from the global separation metrics) ----
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

    pre_nn, pre_aa, pre_na, pre_delta = _sep(S_pre)
    ssm_nn, ssm_aa, ssm_na, ssm_delta = _sep(S_ssm)
    pre_gap = _boundary_stats(S_pre, win_rows, args.boundary_k, video_id)
    ssm_gap = _boundary_stats(S_ssm, win_rows, args.boundary_k, video_id)

    metrics = {
        "video_id": video_id,
        "num_windows": T,
        "num_pure_normal_windows": num_pure_normal,
        "num_transition_windows": num_transition,
        "num_pure_abnormal_windows": num_pure_abnormal,
        "num_other_windows": num_other,
        "anomaly_onset_frame": anomaly_onset_frame,
        "anomaly_onset_window": onset_window,
        "pre_ssm": {
            "NN": pre_nn,
            "AA": pre_aa,
            "NA": pre_na,
            "delta_sep": pre_delta,
            "boundary_gap": pre_gap,
        },
        "ssm_internal": {
            "NN": ssm_nn,
            "AA": ssm_aa,
            "NA": ssm_na,
            "delta_sep": ssm_delta,
            "boundary_gap": ssm_gap,
        },
        "delta_sep_gain": ssm_delta - pre_delta,
        "boundary_gap_gain": ssm_gap - pre_gap,
    }

    # ---- heatmaps (shared color scale; boundary line at onset_pos-0.5) ----
    common_min = float(min(S_pre.min(), S_ssm.min()))
    common_max = float(max(S_pre.max(), S_ssm.max()))
    _draw_heatmap(
        S_pre, onset_pos,
        f"{video_id} — pre-SSM window features "
        f"(cosine similarity, onset window = {onset_window})",
        output_dir / "pre_ssm_cosine_heatmap.png", common_min, common_max,
    )
    _draw_heatmap(
        S_ssm, onset_pos,
        f"{video_id} — SSM internal h "
        f"(cosine similarity, onset window = {onset_window})",
        output_dir / "ssm_internal_cosine_heatmap.png", common_min, common_max,
    )

    # ---- outputs ----
    np.save(output_dir / "pre_ssm_cosine.npy", S_pre)
    np.save(output_dir / "ssm_internal_cosine.npy", S_ssm)
    np.save(output_dir / "pre_ssm_features.npy", X)
    np.save(output_dir / "ssm_internal_features.npy", H)

    with open(output_dir / "window_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["window_index", "start_frame",
                           "valid_end_frame", "binary_gt", "gt_state"],
        )
        writer.writeheader()
        for r in win_rows:
            writer.writerow(r)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- diagnostics ----
    print(
        f"video_id={video_id} "
        f"anomaly_onset_frame={anomaly_onset_frame} "
        f"anomaly_onset_window={onset_window} "
        f"num_windows={T} "
        f"pure_normal={num_pure_normal} transition={num_transition} "
        f"pure_abnormal={num_pure_abnormal} other={num_other}"
    )
    print(
        f"pre_delta_sep={pre_delta:.4f} ssm_delta_sep={ssm_delta:.4f} "
        f"delta_sep_gain={ssm_delta - pre_delta:.4f}"
    )
    print(
        f"pre_boundary_gap={pre_gap} ssm_boundary_gap={ssm_gap} "
        f"boundary_gap_gain={ssm_gap - pre_gap}"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
