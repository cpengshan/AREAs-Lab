#!/bin/bash

REPO_ROOT="Enter repository root path"

CONFIG="${REPO_ROOT}/AREAEnv/area_env/configs/default.yaml"
INPUT_TYPE="elevator_pitch_summary"                # elevator_pitch_summary | deep_dive_summary
MAX_STEPS=5
OUTPUT_DIR="${REPO_ROOT}/results/zero_shot"
AGENT_MODEL="gemini-3.1-pro-preview"         # "gpt-5.4" | "claude-sonnet-4-6" | "gemini-3.1-pro-preview"
AGENT_MODEL_TEMPERATURE=0.4

DATASETS=(
    "alexfabbri/multi_news"
    # "santoshtyss/uk_legislation"
    # "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    # "thu-coai/esconv"
    # "ccdv/arxiv-summarization"
    # "ccdv/govreport-summarization"
    # "ccdv/mediasum"
    # "ccdv/patent-classification"
    # "ccdv/pubmed-summarization"
    # "mrSoul7766/ECTSum"
    # "HuggingFaceFW/fineweb-edu"
    # "HuggingFaceH4/MATH-500"
    # "danidanou/Reuters_Financial_News"
    # "Pavithree/eli5"
    # "FiscalNote/billsum"
    # "Harley-ml/lesswrong"
)

LOG_DIR="${REPO_ROOT}/agent/logs/zero_shot"
mkdir -p "$LOG_DIR"
if [ -n "$1" ]; then
  EXP_ID="$1"
  RESUME_FLAG="--exp_id $EXP_ID"
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Resuming zero_shot Experiment${EXP_ID} (skipping already-done tasks)..."
else
  EXP_ID=1
  while [ -f "${LOG_DIR}/experiment${EXP_ID}.log" ]; do
    EXP_ID=$((EXP_ID + 1))
  done
  RESUME_FLAG=""
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Running zero_shot Experiment${EXP_ID} across all datasets..."
fi

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "=== Dataset: ${DATASET} ==="

  python3 "${REPO_ROOT}/agent/run_agent.py" \
    --config "$CONFIG" \
    --strategy zero_shot \
    --dataset "$DATASET" \
    --input_type "$INPUT_TYPE" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --agent_model "$AGENT_MODEL" \
    --log_file_path "$LOG_FILE" \
    $RESUME_FLAG

done

echo ""
echo "All datasets complete. Log: $LOG_FILE"
