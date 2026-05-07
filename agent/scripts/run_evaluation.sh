#!/bin/bash
#SBATCH --job-name=aunu_evaluation
#SBATCH --output=/local/scratch/zzh2365/AUNU/agent/logs/evaluation/output.log
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --partition=feih100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzh2365@emory.edu

REPO_ROOT="/local/scratch/zzh2365/AUNU"

EVALUATOR_MODEL="gpt-5.4"
REASONING_EFFORT="none"  # none | low | medium | high
WORKERS=10
STRATEGY="user_interaction"          # zero_shot | zero_shot_with_samples_reason | user_interaction ｜ hybrid

# Datasets to evaluate — must match the slugs under agent/results/
# Format: "dataset_name" (the HuggingFace id; slashes become underscores in the results dir)
DATASETS=(
    # "alexfabbri/multi_news"
    # "santoshtyss/uk_legislation"
    # "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    # "thu-coai/esconv"
    # "ccdv/arxiv-summarization"
    "ccdv/govreport-summarization"
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

RESULTS_BASE="${REPO_ROOT}/agent/results"
LOG_DIR="${REPO_ROOT}/agent/logs/evaluation"
mkdir -p "$LOG_DIR"

# Determine log file name (auto-increment to avoid clobbering)
LOG_ID=1
while [ -f "${LOG_DIR}/evaluation${LOG_ID}.log" ]; do
  LOG_ID=$((LOG_ID + 1))
done
LOG_FILE="${LOG_DIR}/evaluation${LOG_ID}.log"

# Optional: pass an experiment ID to target only that experiment across all datasets
#   bash run_evaluation.sh 17   → evaluates only Experiment17 for each dataset
TARGET_EXP="$1"

echo "=========================================="
echo "  AUNU Evaluation Run ${LOG_ID}"
echo "  Strategy : ${STRATEGY}"
echo "  Model    : ${EVALUATOR_MODEL}"
echo "  Reasoning: ${REASONING_EFFORT}"
echo "  Workers  : ${WORKERS}"
echo "  Log      : ${LOG_FILE}"
[ -n "$TARGET_EXP" ] && echo "  Target   : Experiment${TARGET_EXP} only"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
  # Replicate the slug logic from run_agent.py: replace / with _
  DATASET_SLUG="${DATASET//\//_}"
  STRATEGY_DIR="${RESULTS_BASE}/${DATASET_SLUG}/${STRATEGY}"

  if [ ! -d "$STRATEGY_DIR" ]; then
    echo ""
    echo "=== [SKIP] ${DATASET} — no results dir at ${STRATEGY_DIR} ==="
    continue
  fi

  echo ""
  echo "=== Dataset: ${DATASET} ==="

  # Find the experiment directory with the highest numeric ID
  if [ -n "$TARGET_EXP" ]; then
    EXP_DIRS=("${STRATEGY_DIR}/Experiment${TARGET_EXP}")
  else
    LATEST_EXP=$(find "$STRATEGY_DIR" -maxdepth 1 -type d -name "Experiment*" \
      | sort -V | tail -1)
    if [ -z "$LATEST_EXP" ]; then
      echo "  [SKIP] no Experiment dirs found in ${STRATEGY_DIR}"
      continue
    fi
    EXP_DIRS=("$LATEST_EXP")
  fi

  for EXP_DIR in "${EXP_DIRS[@]}"; do
    OUTPUT_JSON="${EXP_DIR}/output.json"
    EVAL_JSON="${EXP_DIR}/eval_results.json"

    if [ ! -f "$OUTPUT_JSON" ]; then
      echo "  [SKIP] ${EXP_DIR} — no output.json"
      continue
    fi

    EXP_NAME=$(basename "$EXP_DIR")
    echo "  Evaluating ${EXP_NAME}..."

    python3 "${REPO_ROOT}/agent/scripts/run_evaluation.py" \
      --input_path       "$OUTPUT_JSON" \
      --output_path      "$EVAL_JSON" \
      --evaluator_model  "$EVALUATOR_MODEL" \
      --reasoning_effort "$REASONING_EFFORT" \
      --workers          "$WORKERS" \
      --resume \
      2>&1 | tee -a "$LOG_FILE"

    STATUS=$?
    if [ $STATUS -ne 0 ]; then
      echo "  [ERROR] ${EXP_NAME} failed (exit $STATUS)" | tee -a "$LOG_FILE"
    else
      echo "  [DONE]  ${EXP_NAME} → ${EVAL_JSON}"
    fi
  done

done

echo ""
echo "=========================================="
echo "All evaluations complete. Log: $LOG_FILE"
echo "=========================================="
