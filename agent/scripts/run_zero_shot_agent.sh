#!/bin/bash
#SBATCH --job-name=aunu_zero_shot
#SBATCH --output=/local/scratch/zzh2365/AUNU/agent/logs/zero_shot/output.log
#SBATCH --mem=2GB
#SBATCH --partition=feih100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzh2365@emory.edu

REPO_ROOT="/local/scratch/zzh2365/AUNU"

CONFIG="${REPO_ROOT}/AUNUEnv/aunu_env/configs/zero_shot.yaml"
INPUT_TYPE="elevator_pitch_summary"                # elevator_pitch_summary | deep_dive_summary

# Datasets to run — comment out any you want to skip
DATASETS=(
    "alexfabbri/multi_news"
    # "ccdv/arxiv-summarization"
    # "ccdv/govreport-summarization"
    # "ccdv/mediasum"
    # "ccdv/patent-classification"
    # "ccdv/pubmed-summarization"
    # "kritsadaK/EDGAR-CORPUS-Financial-Summarization"
    # "rohitsaxena/MovieSum"
    # "santoshtyss/uk_legislation"
    # "HuggingFaceFW/fineweb-edu"
    # "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    # "thu-coai/esconv"
)

LOG_DIR="${REPO_ROOT}/agent/logs/zero_shot"
mkdir -p "$LOG_DIR"
EXP_ID=1
while [ -f "${LOG_DIR}/experiment${EXP_ID}.log" ]; do
  EXP_ID=$((EXP_ID + 1))
done
LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"

echo "Running zero_shot Experiment${EXP_ID} across all datasets..."

for DATASET in "${DATASETS[@]}"; do
  PERSONAS=$(python3 - <<PYEOF
import json, sys
path = "${REPO_ROOT}/AUNUEnv/data/data_synthesized/${DATASET}/synthesized_output.json"
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
    --strategy zero_shot \
    --dataset "$DATASET" \
    --persona $PERSONAS \
    --input_type "$INPUT_TYPE" \
    --log_file_path "$LOG_FILE"

done

echo ""
echo "All datasets complete. Log: $LOG_FILE"
