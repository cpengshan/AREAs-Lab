#!/bin/bash

REPO_ROOT="Enter repository root path"

CONFIG="${REPO_ROOT}/AREAEnv/area_env/configs/default.yaml"
INPUT_TYPE="elevator_pitch_summary"                # elevator_pitch_summary | deep_dive_summary
MAX_ITERATIONS=3                                   # max routing iterations per episode
MAX_STEPS=40
OUTPUT_DIR="${REPO_ROOT}/results/hybrid"
AGENT_MODEL="gemini-3.1-pro-preview"               # "gpt-5.4" | "claude-sonnet-4-6" | "gemini-3.1-pro-preview"

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

LOG_DIR="${REPO_ROOT}/agent/logs/hybrid"
mkdir -p "$LOG_DIR"
if [ -n "$1" ]; then
  EXP_ID="$1"
  RESUME_FLAG="--exp_id $EXP_ID"
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Resuming hybrid Experiment${EXP_ID} (skipping already-done tasks)..."
else
  EXP_ID=1
  while [ -f "${LOG_DIR}/experiment${EXP_ID}.log" ]; do
    EXP_ID=$((EXP_ID + 1))
  done
  RESUME_FLAG=""
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Running hybrid Experiment${EXP_ID} across all datasets..."
fi

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "=== Dataset: ${DATASET} ==="

  python3 "${REPO_ROOT}/agent/run_agent.py" \
    --config "$CONFIG" \
    --strategy hybrid \
    --dataset "$DATASET" \
    --input_type "$INPUT_TYPE" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --agent_model "$AGENT_MODEL" \
    --log_file_path "$LOG_FILE" \
    --communication_habit active \
    $RESUME_FLAG

done

echo ""
echo "All datasets complete. Log: $LOG_FILE"
