#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-ssm lstm stc qformer transformer}"
MAX_HORIZON="${MAX_HORIZON:-64}"
EPOCHS="${EPOCHS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data3/wgq/outputs}"

if [[ "${MAX_HORIZON}" != "64" ]]; then
  echo "Primary long-horizon training requires MAX_HORIZON=64" >&2
  exit 2
fi

for temporal_model in ${MODELS}; do
  log_dir="${OUTPUT_ROOT}/stage1_ucf_long64_${temporal_model}"
  mkdir -p "${log_dir}"
  echo "Training temporal_model=${temporal_model} max_training_horizon=${MAX_HORIZON}"
  python -u pipeline_stage1.py \
    --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
    --train-json /data3/wgq/data/HIVAU-70k/raw_annotations/ucf_database_train.json \
    --video-root /data1/wjq/data/UCF_Crime/training/videos \
    --anomaly-video-root /data1/wjq/data/UCF_Crime/Anomaly-Videos-ALL \
    --feature-cache-root /data3/wgq/data/ucf_visual_cache_v3 \
    --min-pixels 100352 \
    --max-pixels 100352 \
    --max-windows "${MAX_HORIZON}" \
    --limit-training-history-to-chunk \
    --objective score_token \
    --visual-fusion state_spatial_film \
    --temporal-model "${temporal_model}" \
    --temporal-history "${MAX_HORIZON}" \
    --epochs "${EPOCHS}" \
    --log-dir "${log_dir}" \
    --device cuda \
    2>&1 | tee "${log_dir}/train.log"
done
