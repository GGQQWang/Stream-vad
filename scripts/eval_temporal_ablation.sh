#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-ssm lstm stc qformer transformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data3/wgq/outputs}"
PROFILE_WARMUP_WINDOWS="${PROFILE_WARMUP_WINDOWS:-16}"
SSM_CHECKPOINT="${SSM_CHECKPOINT:-/data3/wgq/outputs/stage1_ucf_state_spatial_film/epoch0}"

for temporal_model in ${MODELS}; do
  checkpoint="${OUTPUT_ROOT}/stage1_ucf_temporal_${temporal_model}/epoch0"
  if [[ "${temporal_model}" == "ssm" ]]; then
    checkpoint="${SSM_CHECKPOINT}"
  fi
  infer_dir="${OUTPUT_ROOT}/infer_ucf_temporal_${temporal_model}_epoch0"
  stage_ap_dir="${OUTPUT_ROOT}/stage_ap_ucf_temporal_${temporal_model}_epoch0"
  python -u infer_stage1_ucf.py \
    --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
    --stage1-dir "${checkpoint}" \
    --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
    --video-root /data1/wjq/data/UCF_Crime/testing/videos \
    --feature-cache-root /data3/wgq/data/ucf_visual_cache_v3 \
    --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
    --output-dir "${infer_dir}" \
    --profile-online \
    --profile-warmup-windows "${PROFILE_WARMUP_WINDOWS}" \
    --device cuda
  python -u scripts/eval_stage_ap_ucf.py \
    --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
    --stage1-dir "${checkpoint}" \
    --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
    --video-root /data1/wjq/data/UCF_Crime/testing/videos \
    --feature-cache-root /data3/wgq/data/ucf_visual_cache_v3 \
    --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
    --output-dir "${stage_ap_dir}" \
    --device cuda
done

python -u tools/summarize_temporal_ablation.py \
  --output-root "${OUTPUT_ROOT}" \
  --results-dir "${OUTPUT_ROOT}" \
  --methods ${MODELS}
