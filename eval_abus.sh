#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash eval_abus.sh [single|labeled|full|all]
#
# You can override settings via environment variables, e.g.:
#   MODEL_PATH=/path/to/lingshu MAX_IMAGE_NUM=6 BREASTUS_SEQ_LEN=6 bash eval_abus.sh labeled

MODE="${1:-all}"

DATASETS_PATH="${DATASETS_PATH:-datas/ABUS}"
OUTPUT_BASE="${OUTPUT_BASE:-eval_results/ABUS}"
MODEL_NAME="${MODEL_NAME:-Lingshu-7B}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
USE_VLLM="${USE_VLLM:-False}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Resolve local model path if not provided.
if [[ -z "${MODEL_PATH:-}" ]]; then
  HF_LINGSHU_BASE="${HOME}/.cache/huggingface/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots"
  if [[ -d "${HF_LINGSHU_BASE}" ]]; then
    # Pick the first snapshot (sorted)
    SNAPSHOT_ID="$(ls -1 "${HF_LINGSHU_BASE}" | head -n 1)"
    MODEL_PATH="${HF_LINGSHU_BASE}/${SNAPSHOT_ID}"
  else
    echo "MODEL_PATH not set and no local snapshot found at ${HF_LINGSHU_BASE}."
    echo "Please export MODEL_PATH=/path/to/lingshu-model"
    exit 1
  fi
fi

# Sequence sampling length; 0 = use all frames (still limited by MAX_IMAGE_NUM).
BREASTUS_SEQ_LEN="${BREASTUS_SEQ_LEN:-0}"
BREASTUS_SINGLE_LABELED_N="${BREASTUS_SINGLE_LABELED_N:-5}"
BREASTUS_SINGLE_UNLABELED_N="${BREASTUS_SINGLE_UNLABELED_N:-5}"
BREASTUS_SAMPLE_SEED="${BREASTUS_SAMPLE_SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Optional: custom prompt or CSV name
# export BREASTUS_PROMPT_FILE=/path/to/prompt.txt
# export BREASTUS_REPORTS_CSV=abus_b_g.csv

export BREASTUS_SEQ_LEN
export BREASTUS_SINGLE_LABELED_N
export BREASTUS_SINGLE_UNLABELED_N
export BREASTUS_SAMPLE_SEED
export ATTN_IMPL
export CUDA_VISIBLE_DEVICES
export TEMPERATURE
export TOP_P
export BATCH_SIZE

run_mode () {
  local mode="$1"
  local output_path="${OUTPUT_BASE}_${mode}"
  python eval.py \
    --eval_datasets "BreastUS-${mode}" \
    --datasets_path "${DATASETS_PATH}" \
    --output_path "${output_path}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --use_vllm "${USE_VLLM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_image_num "${MAX_IMAGE_NUM}"
}

case "${MODE}" in
  single)
    run_mode "single"
    ;;
  labeled)
    run_mode "labeled"
    ;;
  full)
    run_mode "full"
    ;;
  all)
    run_mode "single"
    run_mode "labeled"
    run_mode "full"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash eval_abus.sh [single|labeled|full|all]"
    exit 1
    ;;
esac
