#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/output/detector_config_ablation.json}"
MODELS_ROOT="${MODELS_ROOT:-/workspace/storage-shared/nlp/hieplh8/model}"
GPUS="${GPUS:-0,1,2,3}"
PLOT_LOWEST_N="${PLOT_LOWEST_N:-200}"
AGGREGATION="${AGGREGATION:-weighted}"
MIN_SPAN_RECALL="${MIN_SPAN_RECALL:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mapfile -t CONFIG_KEYS < <(
  python3 - "${CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    config = json.load(file)

for key in config:
    print(key)
PY
)

if [[ "$#" -gt 0 ]]; then
  CONFIG_KEYS=("$@")
fi

split_key() {
  local key="$1"
  local base_key="$key"
  local domains=("multihop-reasoning" "science" "logic" "math")

  ABLATION_SIZE=""
  if [[ "${base_key}" =~ ^(.+)-([0-9]+)$ ]]; then
    base_key="${BASH_REMATCH[1]}"
    ABLATION_SIZE="${BASH_REMATCH[2]}"
  fi

  for domain in "${domains[@]}"; do
    local suffix="-${domain}"
    if [[ "${base_key}" == *"${suffix}" ]]; then
      MODEL_NAME="${base_key%"${suffix}"}"
      DOMAIN="${domain}"
      return 0
    fi
  done

  echo "Cannot infer model/domain from ablation config key: ${key}" >&2
  return 1
}

for MODEL_KEY in "${CONFIG_KEYS[@]}"; do
  split_key "${MODEL_KEY}"

  MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
  TEST_PATH="method/calibrate_leakage_detector/data/${MODEL_NAME}/${DOMAIN}/test_detector.jsonl"
  PLOT_DIR="method/calibrate_leakage_detector/output/ablation/${MODEL_NAME}/${MODEL_KEY}/plots"
  RUN_MAX_SEQ_LEN="${MAX_SEQ_LEN}"
  case "${MODEL_NAME}" in
    Qwen3-14B|DeepSeek-R1-Distill-Qwen-14B|Nemo3-Nano-4B*)
      RUN_MAX_SEQ_LEN=4096
      ;;
  esac

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Skipping ${MODEL_KEY}: model path not found: ${MODEL_PATH}" >&2
    continue
  fi
  if [[ ! -f "${TEST_PATH}" ]]; then
    echo "Skipping ${MODEL_KEY}: test file not found: ${TEST_PATH}" >&2
    continue
  fi

  echo "Evaluating ablation ${MODEL_KEY}"
  echo "  model: ${MODEL_PATH}"
  echo "  domain: ${DOMAIN}"
  echo "  ablation_size: ${ABLATION_SIZE:-none}"
  echo "  test : ${TEST_PATH}"
  echo "  plots: ${PLOT_DIR}"
  echo "  max_seq_len: ${RUN_MAX_SEQ_LEN}"

  CUDA_VISIBLE_DEVICES="${GPUS}" python3 method/calibrate_leakage_detector/eval_calibrate_detector.py \
    --model "${MODEL_PATH}" \
    --model-key "${MODEL_KEY}" \
    --test-path "${TEST_PATH}" \
    --detector-config-path "${CONFIG_PATH}" \
    --plot-dir "${PLOT_DIR}" \
    --plot-lowest-n "${PLOT_LOWEST_N}" \
    --aggregation "${AGGREGATION}" \
    --min-span-recall "${MIN_SPAN_RECALL}" \
    --batch-size "${BATCH_SIZE}" \
    --max-seq-len "${RUN_MAX_SEQ_LEN}"
done
