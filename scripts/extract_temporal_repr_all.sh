#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-ssm lstm stc qformer transformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data3/wgq/outputs/temporal_repr_analysis}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
EXTRACT_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRACT_ARGS+=(--overwrite)
elif [[ "${RESUME:-0}" == "1" ]]; then
  EXTRACT_ARGS+=(--resume)
fi

for method in ${MODELS}; do
  if [[ "${method}" == "ssm" ]]; then
    checkpoint="/data3/wgq/outputs/stage1_ucf_state_spatial_film/epoch0"
  else
    checkpoint="/data3/wgq/outputs/stage1_ucf_temporal_${method}/epoch0"
  fi
  python -u scripts/extract_temporal_repr_ucf.py \
    --method "${method}" \
    --stage1-dir "${checkpoint}" \
    --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
    --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
    --video-root /data1/wjq/data/UCF_Crime/testing/videos \
    --feature-cache-root /data3/wgq/data/ucf_visual_cache_v3 \
    --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
    --output-dir "${OUTPUT_ROOT}/${method}" \
    --storage-dtype "${CACHE_DTYPE}" \
    --device cuda \
    "${EXTRACT_ARGS[@]}"
done
