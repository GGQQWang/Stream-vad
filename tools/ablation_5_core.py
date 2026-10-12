"""CPU-only configuration and parsing rules for the A-E ablation."""

from __future__ import annotations

import math
import re


ZERO_SHOT_PROMPT = (
    "Based only on the observed video frames, assign an anomaly score between 0 and 1, "
    "where 0 indicates normal activity and 1 indicates highly anomalous activity. "
    "Output only a single number without any explanation."
)

GROUPS = {
    "A": {"temporal_mode": "none", "semantic": False, "train": False},
    "B": {"temporal_mode": "none", "semantic": False, "train": True},
    "C": {"temporal_mode": "none", "semantic": True, "train": True},
    "D": {"temporal_mode": "ssm", "semantic": False, "train": True},
    "E": {"temporal_mode": "ssm", "semantic": True, "train": True},
}

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_generated_score(text: str) -> tuple[float | None, str | None]:
    value = text.strip()
    if not _NUMBER.fullmatch(value):
        return None, "not_a_single_number"
    score = float(value)
    if not math.isfinite(score):
        return None, "non_finite"
    if not 0 <= score <= 1:
        return None, "out_of_range"
    return score, None


def group_config(group: str, lambda_sem: float) -> dict:
    if group not in GROUPS:
        raise ValueError(f"unsupported ablation group: {group}")
    if lambda_sem <= 0 or not math.isfinite(lambda_sem):
        raise ValueError("lambda_sem must be positive and finite")
    config = dict(GROUPS[group])
    config.update({
        "group": group,
        "lambda_score": 1.0,
        "lambda_sum": lambda_sem if config["semantic"] else 0.0,
        "lambda_world": 0.0,
        "visual_fusion": "state_only",
        "objective": "score_token" if group != "A" else "zero_shot_generative",
    })
    return config
