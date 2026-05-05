#!/bin/bash
#SBATCH --job-name=hybrid_v2
#SBATCH --output=/local/scratch/zzh2365/AUNU/agent/logs/hybrid_v2/output.log
#SBATCH --mem=2GB
#SBATCH --partition=feih100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzh2365@emory.edu

REPO_ROOT="/local/scratch/zzh2365/AUNU"

CONFIG="${REPO_ROOT}/AUNUEnv/aunu_env/configs/default.yaml"
INPUT_TYPE="elevator_pitch_summary"                # elevator_pitch_summary | deep_dive_summary
MAX_ITERATIONS=15                                   # total routing iterations (Phase 1 + Phase 2)
MID_TURN=6                                          # turn at which mid-synthesis fires (end of Phase 1)
MAX_STEPS=40
OUTPUT_DIR="${REPO_ROOT}/agent/results/hybrid_v2"
AGENT_MODEL="claude-haiku-4-5-20251001"

# Datasets to run — format: "data_home_dir:dataset_name"
DATASETS=(
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/alexfabbri/multi_news:alexfabbri/multi_news"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/santoshtyss/uk_legislation:santoshtyss/uk_legislation"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/starmpcc/Asclepius-Synthetic-Clinical-Notes:starmpcc/Asclepius-Synthetic-Clinical-Notes"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/thu-coai/esconv:thu-coai/esconv"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/arxiv-summarization:ccdv/arxiv-summarization"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/govreport-summarization:ccdv/govreport-summarization"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/mediasum:ccdv/mediasum"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/patent-classification:ccdv/patent-classification"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/pubmed-summarization:ccdv/pubmed-summarization"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/mrSoul7766/ECTSum:mrSoul7766/ECTSum"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/HuggingFaceFW/fineweb-edu:HuggingFaceFW/fineweb-edu"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/HuggingFaceH4/MATH-500:HuggingFaceH4/MATH-500"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/danidanou/Reuters_Financial_News:danidanou/Reuters_Financial_News"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/Pavithree/eli5:Pavithree/eli5"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/FiscalNote/billsum:FiscalNote/billsum"
    # "${REPO_ROOT}/AUNUEnv/data/data_synthesized/Harley-ml/lesswrong:Harley-ml/lesswrong"
)

LOG_DIR="${REPO_ROOT}/agent/logs/hybrid_v2"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

if [ -n "$1" ]; then
  EXP_ID="$1"
  RESUME_FLAG="--exp_id $EXP_ID"
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Resuming hybrid_v2 Experiment${EXP_ID} (skipping already-done tasks)..."
else
  EXP_ID=1
  while [ -f "${LOG_DIR}/experiment${EXP_ID}.log" ]; do
    EXP_ID=$((EXP_ID + 1))
  done
  RESUME_FLAG=""
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Running hybrid_v2 Experiment${EXP_ID} across all datasets..."
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
    --strategy hybrid_v2 \
    --dataset "$DATASET" \
    --persona $PERSONAS \
    --input_type "$INPUT_TYPE" \
    --max_iterations "$MAX_ITERATIONS" \
    --mid_turn "$MID_TURN" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --agent_model "$AGENT_MODEL" \
    --log_file_path "$LOG_FILE" \
    --use_v2 \
    --communication_habit neutral \
    $RESUME_FLAG

done

echo ""
echo "All datasets complete. Log: $LOG_FILE"
