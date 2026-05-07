#!/bin/bash
#SBATCH --job-name=aunu_data_analysis_cache
#SBATCH --output=/local/scratch/zzh2365/AUNU/agent/logs/data_analysis_cache/output.log
#SBATCH --mem=2GB
#SBATCH --partition=feih100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzh2365@emory.edu

REPO_ROOT="/local/scratch/zzh2365/AUNU"

OUTPUT_DIR="${REPO_ROOT}/agent/results/data_analysis_cache"

DATASETS=(
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/alexfabbri/multi_news"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/santoshtyss/uk_legislation"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/starmpcc/Asclepius-Synthetic-Clinical-Notes"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/thu-coai/esconv"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/arxiv-summarization"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/govreport-summarization"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/mediasum"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/patent-classification"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/ccdv/pubmed-summarization"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/mrSoul7766/ECTSum"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/HuggingFaceFW/fineweb-edu"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/HuggingFaceH4/MATH-500"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/danidanou/Reuters_Financial_News"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/Pavithree/eli5"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/FiscalNote/billsum"
    "${REPO_ROOT}/AUNUEnv/data/data_synthesized/Harley-ml/lesswrong"
)

LOG_DIR="${REPO_ROOT}/agent/logs/data_analysis_cache"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

if [ -n "$1" ]; then
  EXP_ID="$1"
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Resuming data_analysis_cache Experiment${EXP_ID}..."
else
  EXP_ID=1
  while [ -f "${LOG_DIR}/experiment${EXP_ID}.log" ]; do
    EXP_ID=$((EXP_ID + 1))
  done
  LOG_FILE="${LOG_DIR}/experiment${EXP_ID}.log"
  echo "Running data_analysis_cache Experiment${EXP_ID}..."
fi

OUTPUT_FILE="${OUTPUT_DIR}/experiment${EXP_ID}.json"

echo ""
echo "Datasets selected: ${#DATASETS[@]}"
for D in "${DATASETS[@]}"; do echo "  $D"; done
echo ""

python3 "${REPO_ROOT}/AUNUEnv/scripts/analyze_requirement_categories.py" \
  --model "none" \
  --data-dirs "${DATASETS[@]}" \
  --filename "synthesized_output_2.6.json" \
  --output "$OUTPUT_FILE" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Per-Dataset Category Breakdown ==="
python3 - <<PYEOF
import json, sys

path = "${OUTPUT_FILE}"
try:
    with open(path) as f:
        d = json.load(f)
except Exception as e:
    print(f"ERROR reading {path}: {e}", file=sys.stderr)
    sys.exit(1)

dr = d.get("dataset_results", {})

ORDER = [
    "alexfabbri/multi_news",
    "santoshtyss/uk_legislation",
    "starmpcc/Asclepius-Synthetic-Clinical-Notes",
    "thu-coai/esconv",
    "ccdv/arxiv-summarization",
    "ccdv/govreport-summarization",
    "ccdv/mediasum",
    "ccdv/patent-classification",
    "ccdv/pubmed-summarization",
    "mrSoul7766/ECTSum",
    "HuggingFaceFW/fineweb-edu",
    "HuggingFaceH4/MATH-500",
    "danidanou/Reuters_Financial_News",
    "Pavithree/eli5",
    "FiscalNote/billsum",
    "Harley-ml/lesswrong",
]
# Preserve ORDER for known datasets; append any unexpected ones at the end
ordered_keys = [k for k in ORDER if k in dr] + [k for k in dr if k not in ORDER]

header = f"{'Dataset':<55} {'Tasks':>6} {'Total':>7} {'Data#':>7} {'User#':>7} {'Data%':>7} {'User%':>7}"
print(header)
print("-" * len(header))

for dataset_id in ordered_keys:
    r = dr[dataset_id]
    print(
        f"{dataset_id:<55} "
        f"{r['task_count']:>6} "
        f"{r['total_count']:>7} "
        f"{r['data_specified_count']:>7} "
        f"{r['user_specified_count']:>7} "
        f"{r['data_specified_ratio']*100:>6.1f}% "
        f"{r['user_specified_ratio']*100:>6.1f}%"
    )

total_tasks = sum(r['task_count'] for r in dr.values())
total_units = sum(r['total_count'] for r in dr.values())
total_data  = sum(r['data_specified_count'] for r in dr.values())
total_user  = sum(r['user_specified_count'] for r in dr.values())
print("-" * len(header))
print(
    f"{'TOTAL':<55} "
    f"{total_tasks:>6} "
    f"{total_units:>7} "
    f"{total_data:>7} "
    f"{total_user:>7} "
    f"{total_data/total_units*100 if total_units else 0:>6.1f}% "
    f"{total_user/total_units*100 if total_units else 0:>6.1f}%"
)
PYEOF

echo ""
echo "Results saved to: $OUTPUT_FILE | Log: $LOG_FILE"
