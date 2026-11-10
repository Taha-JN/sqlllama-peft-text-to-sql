#!/usr/bin/env bash
# ==============================================================================
# SQL-LLaMA Distributed Training Launcher
# Supervised Full-Parameter Fine-Tuning with DeepSpeed ZeRO-3
#
# Usage:
#   ./run_training.sh [MODEL_SIZE] [DS_CONFIG] [DATA_PATH] [EPOCHS] [BATCH_SIZE] [GRAD_ACCUM] [PORT]
#
# Examples:
#   ./run_training.sh 7B
#   ./run_training.sh 13B configs/ds_config_sql.json ./datasets/sql_create_dataset_cleaned.json 3 8 16
# ==============================================================================

set -euo pipefail

# Ensure script executes from repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Positional Parameters with Production Defaults
MODEL_SIZE="${1:-7B}"
DS_CONFIG="${2:-configs/ds_config_sql.json}"
DATA_PATH="${3:-./datasets/sql_create_dataset_cleaned.json}"

# Resolve Model Path
if [[ -d "./models_hf/${MODEL_SIZE}" ]]; then
    MODEL_PATH="./models_hf/${MODEL_SIZE}"
elif [[ -d "${MODEL_SIZE}" ]]; then
    MODEL_PATH="${MODEL_SIZE}"
else
    MODEL_PATH="./models_hf/${MODEL_SIZE}"
fi

# Size-dependent Defaults
if [[ "${MODEL_SIZE}" == "13B"* ]]; then
    DEFAULT_EPOCHS=3
    DEFAULT_BATCH_SIZE=8
    DEFAULT_GRAD_ACCUM=16
    DEFAULT_PORT=1211
elif [[ "${MODEL_SIZE}" == "70B"* ]]; then
    DEFAULT_EPOCHS=3
    DEFAULT_BATCH_SIZE=2
    DEFAULT_GRAD_ACCUM=64
    DEFAULT_PORT=1299
else
    DEFAULT_EPOCHS=5
    DEFAULT_BATCH_SIZE=16
    DEFAULT_GRAD_ACCUM=32
    DEFAULT_PORT=1255
fi

NUM_EPOCHS="${4:-${DEFAULT_EPOCHS}}"
BATCH_SIZE="${5:-${DEFAULT_BATCH_SIZE}}"
GRAD_ACCUM="${6:-${DEFAULT_GRAD_ACCUM}}"
MASTER_PORT="${7:-${DEFAULT_PORT}}"
OUTPUT_DIR="./output_sqlAlpaca${MODEL_SIZE}/"

echo "=================================================================="
echo " Starting SQL-LLaMA Distributed Fine-Tuning"
echo "=================================================================="
echo " Repository Root        : ${REPO_ROOT}"
echo " Model Target           : ${MODEL_SIZE} (${MODEL_PATH})"
echo " Dataset Path           : ${DATA_PATH}"
echo " DeepSpeed Config       : ${DS_CONFIG}"
echo " Training Epochs        : ${NUM_EPOCHS}"
echo " Per-device Batch Size  : ${BATCH_SIZE}"
echo " Gradient Accumulation  : ${GRAD_ACCUM}"
echo " Distributed Master Port: ${MASTER_PORT}"
echo " Output Checkpoint Dir  : ${OUTPUT_DIR}"
echo "=================================================================="

# Check prerequisites
if [[ ! -f "${DS_CONFIG}" ]]; then
    echo "ERROR: DeepSpeed configuration '${DS_CONFIG}' not found!" >&2
    exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "WARNING: Dataset '${DATA_PATH}' not found. Please ensure data preparation has been run." >&2
fi

torchrun --master_port="${MASTER_PORT}" src/sqlllama/training/trainer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size $((BATCH_SIZE * 2)) \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --deepspeed "${DS_CONFIG}" \
    --tf32 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1
