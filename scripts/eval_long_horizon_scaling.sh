#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-ssm lstm stc qformer transformer}"
HORIZONS="${HORIZONS:-4 8 16 32 64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data3/wgq/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${OUTPUT_ROOT}/long_horizon_scaling}"
PROFILE_WARMUP_WINDOWS="${PROFILE_WARMUP_WINDOWS:-16}"

for temporal_model in ${MODELS}; do
  checkpoint="${OUTPUT_ROOT}/stage1_ucf_long64_${temporal_model}/epoch0"
  for horizon in ${HORIZONS}; do
    result_dir="${RESULTS_ROOT}/${temporal_model}/T${horizon}"
    mkdir -p "${result_dir}"
    python -u scripts/eval_long_horizon_ucf.py \
      --method "${temporal_model}" \
      --history-horizon "${horizon}" \
      --model-path /data3/wgq/models/Qwen2-VL-7B-Instruct \
      --stage1-dir "${checkpoint}" \
      --test-manifest /data3/wgq/data/ucf_original_cache_manifests/ucf_original_test_cache_manifest.json \
      --video-root /data1/wjq/data/UCF_Crime/testing/videos \
      --feature-cache-root /data3/wgq/data/ucf_visual_cache_v3 \
      --gt-root /data1/wjq/data/UCF_Crime/testing/gt_labels \
      --output-dir "${result_dir}" \
      --profile-warmup-windows "${PROFILE_WARMUP_WINDOWS}" \
      --device cuda \
      2>&1 | tee "${result_dir}/eval.log"
  done
done

python -u tools/summarize_long_horizon_scaling.py \
  --results-root "${RESULTS_ROOT}" \
  --output-dir "${RESULTS_ROOT}" \
  --methods ${MODELS} \
  --horizons ${HORIZONS}
