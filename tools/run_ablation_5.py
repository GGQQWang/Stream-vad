"""Orchestrate A-E with independent, auditable UCF/XD runs (no GPU imports)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ablation_5_core import GROUPS, group_config


DATASETS = ("ucf", "xd")
DEFAULT_OUTPUT = Path("/data3/wgq/outputs/ablation_5")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_groups(args) -> list[str]:
    if args.group and args.groups:
        raise ValueError("--group and --groups are mutually exclusive")
    groups = [args.group] if args.group else args.groups.split(",") if args.groups else list(GROUPS)
    if not groups or any(group not in GROUPS for group in groups) or len(set(groups)) != len(groups):
        raise ValueError("groups must be unique members of A,B,C,D,E")
    return groups


def load_config(path: Path, *, groups: list[str], train: bool, evaluate: bool,
                dry_run: bool, datasets: tuple[str, ...] = DATASETS) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required_common = ("model_path", "seed", "epochs", "lr", "batch_size", "grad_accum", "lambda_sem", "pixels")
    missing = [key for key in required_common if key not in config or config[key] in ("", None)]
    if missing:
        raise ValueError(f"ablation config missing common values: {missing}")
    if int(config["pixels"]) != 100352:
        raise ValueError("current strict inference uses 100352 pixels; training config must match")
    if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1 or int(config["grad_accum"]) < 1:
        raise ValueError("epochs/batch_size/grad_accum must be positive")
    if float(config["lr"]) <= 0:
        raise ValueError("lr must be positive")
    for group in groups:
        group_config(group, float(config["lambda_sem"]))
    needs_training_data = train and any(group != "A" for group in groups)
    for dataset in datasets:
        details = config.get(dataset)
        if not isinstance(details, dict):
            raise ValueError(f"ablation config missing {dataset} object")
        required = []
        if needs_training_data:
            required.extend(("train_json", "train_video_root", "feature_cache_root"))
        if evaluate:
            required.extend(("test_manifest", "test_video_root", "test_gt_root"))
            if any(group != "A" for group in groups):
                required.append("feature_cache_root")
        missing = [key for key in required if not details.get(key)]
        if missing:
            raise ValueError(f"ablation config {dataset} missing: {missing}")
        if not dry_run:
            files = {"train_json", "test_manifest"}
            for key in required:
                candidate = Path(details[key])
                if key in files and not candidate.is_file():
                    raise FileNotFoundError(f"{dataset}.{key}: {candidate}")
                if key not in files and not candidate.is_dir():
                    raise NotADirectoryError(f"{dataset}.{key}: {candidate}")
    if not dry_run and not Path(config["model_path"]).is_dir():
        raise NotADirectoryError(f"model_path: {config['model_path']}")
    return config


def latest_checkpoint(train_dir: Path) -> tuple[int, Path] | None:
    candidates = []
    for path in train_dir.glob("epoch*/train_state.pt"):
        epoch_text = path.parent.name.removeprefix("epoch")
        if epoch_text.isdigit() and (path.parent / "lora_adapter").is_dir():
            candidates.append((int(epoch_text), path.parent))
    return max(candidates, default=None)


def next_eval_dir(dataset_root: Path, *, resume: bool) -> tuple[Path, bool]:
    existing = [dataset_root / "eval"] + sorted(dataset_root.glob("eval_retry_*"))
    for path in reversed(existing):
        if (path / "metrics.json").is_file():
            metrics = json.loads((path / "metrics.json").read_text())
            if int(metrics.get("num_failed_videos", 1)) == 0 and metrics.get("global_standard_auc") is not None:
                if resume:
                    return path, True
                raise FileExistsError(f"evaluation already exists: {path}; use --resume or a new output root")
    if not (dataset_root / "eval").exists():
        return dataset_root / "eval", False
    if not resume:
        raise FileExistsError(f"partial evaluation exists: {dataset_root / 'eval'}; use --resume")
    index = 1
    while (dataset_root / f"eval_retry_{index}").exists():
        index += 1
    return dataset_root / f"eval_retry_{index}", False


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: list[str], *, dataset_root: Path, stage: str, dry_run: bool, env: dict) -> None:
    rendered = shlex.join(cmd)
    print(f"[{stage}] {rendered}", flush=True)
    if dry_run:
        return
    dataset_root.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "command": cmd, "started_at": utc_now()}
    with (dataset_root / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    status_path = dataset_root / "status.json"
    save_json(status_path, {"state": "running", **record})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = dataset_root / f"{stage}_{stamp}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    state = "complete" if result.returncode == 0 else "failed"
    save_json(status_path, {"state": state, **record, "finished_at": utc_now(),
                            "exit_code": result.returncode, "log_path": str(log_path)})
    print(f"[{stage}] {state}: {log_path}", flush=True)
    if result.returncode:
        raise RuntimeError(f"{stage} failed (exit {result.returncode}); inspect {log_path}")


def train_command(config: dict, details: dict, group: str, train_dir: Path, device: str,
                  resume_dir: Path | None, smoke_updates: int) -> list[str]:
    scheme = group_config(group, float(config["lambda_sem"]))
    epochs = 1 if smoke_updates else int(config["epochs"])
    cmd = [sys.executable, "-u", "pipeline_stage1.py",
           "--model-path", config["model_path"],
           "--train-json", details["train_json"],
           "--video-root", details["train_video_root"],
           "--feature-cache-root", details["feature_cache_root"],
           "--log-dir", str(train_dir),
           "--epochs", str(epochs), "--lr", str(config["lr"]),
           "--batch-size", str(config["batch_size"]), "--grad-accum", str(config["grad_accum"]),
           "--lora-r", "8", "--lora-alpha", "16", "--d-ssm", "256",
           "--frames-per-clip", "16", "--sample-interval", "3", "--max-windows", "8",
           "--min-pixels", str(config["pixels"]), "--max-pixels", str(config["pixels"]),
           "--objective", "score_token", "--visual-fusion", "state_only",
           "--ablation-group", group, "--lambda-score", "1",
           "--lambda-sum", str(scheme["lambda_sum"]), "--lambda-world", "0",
           "--world-warmup-steps", "0", "--seed", str(config["seed"]),
           "--device", device, "--save-every", "1"]
    if details.get("anomaly_video_root"):
        cmd += ["--anomaly-video-root", details["anomaly_video_root"]]
    if resume_dir is not None:
        cmd += ["--resume", str(resume_dir)]
    if smoke_updates:
        cmd += ["--max-train-updates", str(smoke_updates)]
    return cmd


def eval_command(config: dict, details: dict, group: str, ckpt_dir: Path | None,
                 eval_dir: Path, device: str, smoke_video_id: str) -> list[str]:
    script = "tools/infer_zero_shot_ablation.py" if group == "A" else "infer_stage1_ucf.py"
    cmd = [sys.executable, "-u", script,
           "--model-path", config["model_path"],
           "--test-manifest", details["test_manifest"],
           "--video-root", details["test_video_root"],
           "--gt-root", details["test_gt_root"],
           "--output-dir", str(eval_dir), "--device", device, "--strict"]
    if group != "A":
        cmd += ["--stage1-dir", str(ckpt_dir),
                "--feature-cache-root", details["feature_cache_root"],
                "--inference-max-windows", "1", "--official-gt-only"]
    if smoke_video_id:
        cmd += ["--video-id", smoke_video_id]
    return cmd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/ablation_5.config.json")
    parser.add_argument("--group", choices=tuple(GROUPS))
    parser.add_argument("--groups", help="comma-separated subset, e.g. B,C,D,E")
    parser.add_argument("--dataset", choices=("both", *DATASETS), default="both",
                        help="select UCF only, XD only, or both (default)")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-updates", type=int, default=0,
                        help="separate output root required; limit training to N updates and evaluate one configured video")
    args = parser.parse_args(argv)
    if args.train_only and args.eval_only:
        parser.error("--train-only and --eval-only are mutually exclusive")
    if args.smoke_updates < 0:
        parser.error("--smoke-updates must be nonnegative")
    if args.smoke_updates and args.output_root == DEFAULT_OUTPUT:
        parser.error("smoke tests require a separate --output-root")
    groups = selected_groups(args)
    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    train, evaluate = not args.eval_only, not args.train_only
    config = load_config(
        args.config, groups=groups, train=train, evaluate=evaluate,
        dry_run=args.dry_run, datasets=datasets,
    )
    if args.smoke_updates and evaluate:
        for dataset in datasets:
            if not config[dataset].get("smoke_video_id"):
                parser.error(f"{dataset}.smoke_video_id is required for smoke evaluation")
    env = dict(os.environ)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    print(f"Git commit={commit} dirty={dirty} groups={','.join(groups)} "
          f"datasets={','.join(datasets)} output={args.output_root}")
    for group in groups:
        for dataset in datasets:
            details = config[dataset]
            dataset_root = args.output_root / group / dataset
            train_dir = dataset_root / "train"
            target_epoch = 0 if args.smoke_updates else int(config["epochs"]) - 1
            target_ckpt = train_dir / f"epoch{target_epoch}"
            if not args.dry_run:
                dataset_root.mkdir(parents=True, exist_ok=True)
                snapshot = {"group_config": group_config(group, float(config["lambda_sem"])),
                            "dataset": dataset, "dataset_config": details, "common_config": {
                                key: value for key, value in config.items() if key not in DATASETS},
                            "git_commit": commit, "git_dirty": dirty,
                            "smoke_updates": args.smoke_updates}
                snapshot_path = dataset_root / "config.json"
                if snapshot_path.is_file():
                    previous = json.loads(snapshot_path.read_text())
                    if previous != snapshot:
                        raise ValueError(f"run config changed at {snapshot_path}; use a new output root")
                else:
                    save_json(snapshot_path, snapshot)
            if train and group != "A":
                if (target_ckpt / "train_state.pt").is_file():
                    if not args.resume and not args.dry_run:
                        raise FileExistsError(f"checkpoint already exists: {target_ckpt}; use --resume")
                    print(f"[train] checkpoint already complete: {target_ckpt}")
                else:
                    latest = latest_checkpoint(train_dir) if args.resume else None
                    if latest and latest[0] >= target_epoch:
                        raise RuntimeError(f"invalid final checkpoint: {latest[1]}")
                    if not args.resume and train_dir.exists() and any(train_dir.glob("epoch*")):
                        raise FileExistsError(f"partial checkpoints exist: {train_dir}; use --resume")
                    command = train_command(config, details, group, train_dir, args.device,
                                            latest[1] if latest else None, args.smoke_updates)
                    run_command(command, dataset_root=dataset_root, stage="train", dry_run=args.dry_run, env=env)
                    if not args.dry_run and not (target_ckpt / "train_state.pt").is_file():
                        raise RuntimeError(f"training exited without expected final checkpoint: {target_ckpt}")
            if evaluate:
                if group != "A" and not args.dry_run and not (target_ckpt / "train_state.pt").is_file():
                    raise FileNotFoundError(f"eval-only requires checkpoint: {target_ckpt}")
                eval_dir, already_done = (dataset_root / "eval", False) if args.dry_run else next_eval_dir(
                    dataset_root, resume=args.resume,
                )
                if already_done:
                    print(f"[eval] existing complete result: {eval_dir}")
                    continue
                smoke_video_id = details.get("smoke_video_id", "") if args.smoke_updates else ""
                command = eval_command(config, details, group, target_ckpt if group != "A" else None,
                                       eval_dir, args.device, smoke_video_id)
                run_command(command, dataset_root=dataset_root, stage="eval", dry_run=args.dry_run, env=env)
    if evaluate and not args.dry_run:
        command = [sys.executable, "tools/summarize_ablation_5.py", "--output-root", str(args.output_root),
                   "--dataset", args.dataset]
        run_command(command, dataset_root=args.output_root, stage="summarize", dry_run=False, env=env)


if __name__ == "__main__":
    main()
