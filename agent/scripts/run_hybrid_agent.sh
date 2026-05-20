#!/bin/bash

REPO_ROOT="Enter repository root path"

CONFIG="${REPO_ROOT}/AREAEnv/area_env/configs/default.yaml"
INPUT_TYPE="elevator_pitch_summary"                # elevator_pitch_summary | deep_dive_summary
MAX_ITERATIONS=3                                  # max routing iterations per episode
MAX_STEPS=40
OUTPUT_DIR="${REPO_ROOT}/results/hybrid"
AGENT_MODEL="gemini-3.1-pro-preview"                 # "gpt-5.4" | "claude-sonnet-4-6" | "gemini-3.1-pro-preview"

# Datasets to run — format: "data_home_dir:dataset_name"
DATASETS=(
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/alexfabbri/multi_news:alexfabbri/multi_news"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/santoshtyss/uk_legislation:santoshtyss/uk_legislation"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/starmpcc/Asclepius-Synthetic-Clinical-Notes:starmpcc/Asclepius-Synthetic-Clinical-Notes"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/thu-coai/esconv:thu-coai/esconv"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/ccdv/arxiv-summarization:ccdv/arxiv-summarization"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/ccdv/govreport-summarization:ccdv/govreport-summarization"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/ccdv/mediasum:ccdv/mediasum"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/ccdv/patent-classification:ccdv/patent-classification"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/ccdv/pubmed-summarization:ccdv/pubmed-summarization"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/mrSoul7766/ECTSum:mrSoul7766/ECTSum"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/HuggingFaceFW/fineweb-edu:HuggingFaceFW/fineweb-edu"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/HuggingFaceH4/MATH-500:HuggingFaceH4/MATH-500"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/danidanou/Reuters_Financial_News:danidanou/Reuters_Financial_News"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/Pavithree/eli5:Pavithree/eli5"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/FiscalNote/billsum:FiscalNote/billsum"
    "${REPO_ROOT}/AREAEnv/data/data_synthesized/Harley-ml/lesswrong:Harley-ml/lesswrong"
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


for ENTRY in "${DATASETS[@]}"; do
  DATA_DIR="${ENTRY%%:*}"
  DATASET="${ENTRY##*:}"

  PERSONAS=$(python3 - <<PYEOF
import json, sys
path = "${DATA_DIR}/synthesized_output_2.6.json"
try:
    with open(path) as f:
        d = json.load(f)
    ids = sorted(int(k.split("_")[1]) for k in d if k.startswith("user_"))
    print(" ".join(str(i) for i in ids))
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
  )

  if [ -z "$PERSONAS" ]; then
    echo "ERROR: Could not resolve personas for ${DATASET}, skipping." >&2
    continue
  fi

  echo ""
  echo "=== Dataset: ${DATASET} | Personas: ${PERSONAS} ==="

  python3 "${REPO_ROOT}/agent/run_agent.py" \
    --config "$CONFIG" \
    --strategy hybrid \
    --dataset "$DATASET" \
    --persona $PERSONAS \
    --input_type "$INPUT_TYPE" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --agent_model "$AGENT_MODEL" \
    --log_file_path "$LOG_FILE" \
    --use_v2 \
    --communication_habit active \
    \
    $RESUME_FLAG

done

echo ""
echo "All datasets complete. Log: $LOG_FILE"
