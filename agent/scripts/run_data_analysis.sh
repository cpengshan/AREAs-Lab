#!/bin/bash
#SBATCH --job-name=aunu_data_analysis
#SBATCH --output=/local/scratch/zzh2365/AUNU/agent/logs/data_analysis/output.log
#SBATCH --mem=2GB
#SBATCH --partition=feih100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzh2365@emory.edu

REPO_ROOT="/local/scratch/zzh2365/AUNU"

EVALUATOR_MODEL="gpt-5.4"
OUTPUT_DIR="${REPO_ROOT}/agent/results/data_analysis"

# Datasets to run — format: "data_home_dir:dataset_name"
# data_home_dir: folder containing the synthesized_output.json for that dataset
# dataset_name:  HuggingFace dataset id (used only as a label here)
DATASETS=(
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/alexfabbri/multi_news:alexfabbri/multi_news"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/santoshtyss/uk_legislation:santoshtyss/uk_legislation"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/starmpcc/Asclepius-Synthetic-Clinical-Notes:starmpcc/Asclepius-Synthetic-Clinical-Notes"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/thu-coai/esconv:thu-coai/esconv"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/arxiv-summarization:ccdv/arxiv-summarization"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/govreport-summarization:ccdv/govreport-summarization"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/mediasum:ccdv/mediasum"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/patent-classification:ccdv/patent-classification"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/pubmed-summarization:ccdv/pubmed-summarization"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/mrSoul7766/ECTSum:mrSoul7766/ECTSum"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/HuggingFaceFW/fineweb-edu:HuggingFaceFW/fineweb-edu"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/HuggingFaceH4/MATH-500:HuggingFaceH4/MATH-500"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/danidanou/Reuters_Financial_News:danidanou/Reuters_Financial_News"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/Pavithree/eli5:Pavithree/eli5"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/FiscalNote/billsum:FiscalNote/billsum"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/Harley-ml/lesswrong:Harley-ml/lesswrong"
)

LOG_DIR="${REPO_ROOT}/agent/logs/data_analysis"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

if [ -n "$1" ]; then
  EXP_ID="$1"
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Resuming data_analysis Experiment${EXP_ID}..."
else
  EXP_ID=1
  while [ -f "${LOG_DIR}/experiment${EXP_ID}.log" ]; do
    EXP_ID=$((EXP_ID + 1))
  done
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Running data_analysis Experiment${EXP_ID}..."
fi

OUTPUT_FILE="${OUTPUT_DIR}/experiment${EXP_ID}.json"

# Collect the data_home_dir from each DATASETS entry
DATA_DIRS=()
for ENTRY in "${DATASETS[@]}"; do
  DATA_DIRS+=("${ENTRY%%:*}")
done

echo ""
echo "Datasets selected: ${#DATA_DIRS[@]}"
for D in "${DATA_DIRS[@]}"; do echo "  $D"; done
echo ""

python3 "${REPO_ROOT}/AUNUEnv/scripts/analyze_requirement_categories.py" \
  --model "$EVALUATOR_MODEL" \
  --data-dirs "${DATA_DIRS[@]}" \
  --filename "synthesized_output_2.6.json" \
  --output "$OUTPUT_FILE" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Data analysis complete. Results: $OUTPUT_FILE | Log: $LOG_FILE"
