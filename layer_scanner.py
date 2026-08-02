"""Layer-depth scanner: find the smallest K with full-model AUC.

After Stage-1 training (SSM + full LLM), freeze all weights, then
evaluate the score head on top of the first K transformer layers.
Pick the smallest K whose AUC is within `tol` of the full-model AUC.

Usage:
    python -m compression.layer_scanner \
        --ckpt path/to/stage1.pt \
        --data path/to/ucf_database_test.json \
        --video-root path/to/videos/test \
        --K-list 4,6,8,10,12,16
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from shallow_llm import ShallowLLM, _get_llm_layers


def _load_model(ckpt_path: str):
    """Minimal loader – override in real usage with your actual model class."""
    # Placeholder: replace with your real Stage-1 model loading logic
    raise NotImplementedError(
        "Replace this function with your actual model-loading code. "
        "It should return (model, full_llm) where model is your "
        "trained Stage-1 checkpoint and full_llm is the underlying "
        "Qwen2VLForConditionalGeneration."
    )


def evaluate_k(
    full_llm: torch.nn.Module,
    K: int,
    dataloader,
    device: torch.device,
) -> float:
    """Compute window-level AUC for a given K."""
    shallow = ShallowLLM(full_llm, K=K).to(device)
    shallow.eval()

    all_scores: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"K={K}", leave=False):
            # batch is expected to provide SSM tokens + labels
            ssm_tokens = batch["ssm_tokens"].to(device)   # [B, T, D]
            labels = batch["labels"].to(device)            # [B, T]

            scores, _ = shallow(ssm_tokens)               # [B, T]

            # flatten batch + window dims
            all_scores.extend(scores.flatten().cpu().tolist())
            all_labels.extend(labels.flatten().cpu().tolist())

    auc = roc_auc_score(all_labels, all_scores)
    return auc


def scan(
    full_llm: torch.nn.Module,
    dataloader,
    K_list: List[int],
    device: torch.device,
    tol: float = 0.005,
) -> Dict[int, float]:
    """Return AUC for each K and identify the sweet spot."""
    full_auc = evaluate_k(full_llm, K=len(full_llm.model.language_model.layers
                                         if hasattr(full_llm.model, "language_model")
                                         else full_llm.model.layers),
                          dataloader=dataloader, device=device)
    print(f"Full model AUC: {full_auc:.4f}")

    results = {}
    best_K = None
    for K in sorted(K_list):
        auc = evaluate_k(full_llm, K, dataloader, device)
        results[K] = auc
        gap = full_auc - auc
        tag = " <--" if (best_K is None and gap <= tol) else ""
        print(f"  K={K:2d}  AUC={auc:.4f}  (gap {gap:.4f}){tag}")
        if best_K is None and gap <= tol:
            best_K = K

    if best_K is not None:
        print(f"\nRecommended K = {best_K} (first K with gap ≤ {tol})")
    else:
        print(f"\nNo K meets the {tol} tolerance. Consider raising tol.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Stage-1 checkpoint")
    parser.add_argument("--data", required=True, help="JSON annotation file")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--K-list", default="4,6,8,10,12,16")
    parser.add_argument("--tol", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Replace with your real loader
    _, full_llm = _load_model(args.ckpt)
    full_llm = full_llm.to(device).eval()

    K_list = [int(x) for x in args.K_list.split(",")]

    # Replace with your real dataloader
    # dataloader = build_val_loader(args.data, args.video_root)
    raise NotImplementedError(
        "Replace _load_model() and build a DataLoader that yields "
        "dicts with keys 'ssm_tokens' [B,T,D] and 'labels' [B,T]."
    )

    # scan(full_llm, dataloader, K_list, device, tol=args.tol)


if __name__ == "__main__":
    main()
