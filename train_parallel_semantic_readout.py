"""Train a frozen-backbone parallel semantic readout from summary hidden states."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hivau_dataset import HIVAUDataset, hivau_collate
from hivau_sampler import VideoChunkSampler
from infer_stage1_ucf import (
    FRAMES_PER_CLIP,
    MAX_PIXELS,
    MAX_WINDOWS,
    MIN_PIXELS,
    SAMPLE_INTERVAL,
    _find_embed,
    load_stage1_model,
)
from parallel_semantic_readout import (
    ParallelSemanticReadout,
    decode_parallel_tokens,
    encode_parallel_semantic_targets,
    freeze_all_except_parallel_readout,
    parallel_semantic_loss,
)
from stage1_streaming import collect_summary_triggers


class ParallelSemanticExperiment(nn.Module):
    def __init__(self, stage1_model: nn.Module, parallel_readout: ParallelSemanticReadout) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.parallel_readout = parallel_readout


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stage1_dir_from_checkpoint(path: str | Path) -> Path:
    p = Path(path)
    return p.parent if p.is_file() else p


def _print_freeze_report(report: dict) -> None:
    print(f"Total parameters: {report['total_parameters']}")
    print(f"Frozen parameters: {report['frozen_parameters']}")
    print(f"Trainable parameters: {report['trainable_parameters']}")
    print("Trainable parameter names:")
    for name in report["trainable_parameter_names"]:
        print(f"  {name}")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Stage-1 output dir or train_state.pt")
    parser.add_argument("--train-data", required=True, help="training annotation JSON")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-root", default="")
    parser.add_argument("--num-semantic-queries", type=int, default=16)
    parser.add_argument("--semantic-decoder-dim", type=int, default=256)
    parser.add_argument("--semantic-num-heads", type=int, default=8)
    parser.add_argument("--semantic-num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--timing-log-every", type=int, default=20)
    args = parser.parse_args()

    _set_seed(args.seed)
    args.device = torch.device(args.device)
    args.stage1_dir = str(_stage1_dir_from_checkpoint(args.checkpoint))

    stage1_model, _, tokenizer, dtype, _ = load_stage1_model(args)
    embed_fn = _find_embed(stage1_model.qwen)
    embed_weight = embed_fn.weight
    vocab_size = int(embed_weight.shape[0])
    hidden_size = int(embed_weight.shape[1])

    readout = ParallelSemanticReadout(
        hidden_size,
        vocab_size,
        num_queries=args.num_semantic_queries,
        decoder_dim=args.semantic_decoder_dim,
        num_heads=args.semantic_num_heads,
        num_layers=args.semantic_num_layers,
        use_output_projection=True,
    ).to(args.device)
    experiment = ParallelSemanticExperiment(stage1_model, readout).to(args.device)
    freeze_report = freeze_all_except_parallel_readout(experiment, "parallel_readout")
    _print_freeze_report(freeze_report)
    experiment.stage1_model.eval()
    experiment.parallel_readout.train()

    dataset_start = time.perf_counter()
    dataset = HIVAUDataset(
        args.train_data,
        args.video_root,
        total_sampled_frames=FRAMES_PER_CLIP,
        sample_interval=SAMPLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        feature_cache_root=args.feature_cache_root,
        feature_cache_model_id=args.model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        validate_feature_cache_on_init=False,
        profile_cache_io=True,
    )
    print(f"parallel_semantic_timing: dataset_init_seconds={time.perf_counter() - dataset_start:.3f}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=VideoChunkSampler(dataset.samples, shuffle=True),
        num_workers=args.num_workers,
        collate_fn=hivau_collate,
    )
    optimizer = torch.optim.AdamW(experiment.parallel_readout.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    timing_batch_count = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Parallel semantic epoch {epoch}")
        last_step_end = time.perf_counter()
        for batch in pbar:
            timing_batch_count += 1
            data_wait_sec = time.perf_counter() - last_step_end
            if "features" not in batch:
                raise RuntimeError("parallel semantic training requires feature cache")
            valid_mask = batch["valid_mask"].to(args.device)
            _sync_if_cuda(args.device)
            stage1_start = time.perf_counter()
            with torch.no_grad():
                features = batch["features"].to(device=args.device, dtype=dtype)
                state_emb, _, _, _ = experiment.stage1_model.encode_window_features(
                    features,
                    valid_mask,
                    batch["video_id"],
                    {},
                    training=False,
                )
                triggers, _ = collect_summary_triggers(batch, batch["valid_mask"])
                if not triggers:
                    _sync_if_cuda(args.device)
                    if args.timing_log_every > 0 and timing_batch_count % args.timing_log_every == 0:
                        print(
                            "parallel_semantic_timing: "
                            f"step={global_step} "
                            f"batch={timing_batch_count} "
                            f"data_wait_cache_sec={data_wait_sec:.3f} "
                            "frozen_stage1_forward_sec="
                            f"{time.perf_counter() - stage1_start:.3f} "
                            "parallel_readout_forward_backward_sec=0.000 "
                            "skipped_no_summary_trigger=1"
                        )
                    last_step_end = time.perf_counter()
                    continue
                trigger_b = torch.tensor([t[0] for t in triggers], dtype=torch.long, device=args.device)
                trigger_w = torch.tensor([t[1] for t in triggers], dtype=torch.long, device=args.device)
                trigger_states = state_emb[trigger_b, trigger_w]
                sum_hidden = experiment.stage1_model.forward_summary_query_hidden(
                    trigger_states, embed_fn,
                )
                z_sem = sum_hidden.detach()
            _sync_if_cuda(args.device)
            stage1_sec = time.perf_counter() - stage1_start

            summary_texts = [str(t[2]["text"]) for t in triggers]
            targets = encode_parallel_semantic_targets(
                tokenizer,
                summary_texts,
                max_tokens=args.num_semantic_queries,
                device=args.device,
            )
            _sync_if_cuda(args.device)
            readout_start = time.perf_counter()
            logits = experiment.parallel_readout(z_sem)
            loss, info = parallel_semantic_loss(
                logits,
                targets,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _sync_if_cuda(args.device)
            readout_sec = time.perf_counter() - readout_start
            global_step += 1

            pred = logits.argmax(dim=-1)
            preview = decode_parallel_tokens(tokenizer, pred[:1])
            pbar.set_postfix({
                "parallel_sem_loss": f"{loss.item():.4f}",
                "token_acc": f"{info['token_accuracy']:.3f}",
                "avg_len": f"{info['avg_pred_length']:.2f}",
            })
            if global_step % 100 == 0:
                print(f"GT: {summary_texts[0]}")
                print(f"Pred: {preview[0] if preview else ''}")
            if args.timing_log_every > 0 and timing_batch_count % args.timing_log_every == 0:
                print(
                    "parallel_semantic_timing: "
                    f"step={global_step} "
                    f"batch={timing_batch_count} "
                    f"data_wait_cache_sec={data_wait_sec:.3f} "
                    f"frozen_stage1_forward_sec={stage1_sec:.3f} "
                    f"parallel_readout_forward_backward_sec={readout_sec:.3f}"
                )
            if args.max_steps and global_step >= args.max_steps:
                break
            last_step_end = time.perf_counter()
        if args.max_steps and global_step >= args.max_steps:
            break

    torch.save(
        {
            "parallel_readout": experiment.parallel_readout.state_dict(),
            "config": {
                "input_dim": hidden_size,
                "vocab_size": vocab_size,
                "num_semantic_queries": args.num_semantic_queries,
                "decoder_dim": args.semantic_decoder_dim,
                "num_heads": args.semantic_num_heads,
                "num_layers": args.semantic_num_layers,
                "uses_frozen_lm_head": False,
            },
            "freeze_report": freeze_report,
        },
        output_dir / "parallel_semantic_readout.pt",
    )
    with open(output_dir / "freeze_report.json", "w") as f:
        json.dump(freeze_report, f, indent=2)
    print(f"Saved parallel semantic readout to {output_dir}")


if __name__ == "__main__":
    main()
