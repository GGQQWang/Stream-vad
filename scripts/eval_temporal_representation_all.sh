#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-ssm lstm stc qformer transformer}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-/data3/wgq/outputs/temporal_repr_analysis}"

python -u scripts/eval_temporal_representation.py \
  --analysis-root "${ANALYSIS_ROOT}" \
  --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
  --output-dir "${ANALYSIS_ROOT}" \
  --methods ${MODELS} \
  --k-values 1 2 4 8 16 32 64
