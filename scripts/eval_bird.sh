#!/usr/bin/env bash
# BIRD evaluation pipeline (vLLM n=30 → vav majority voting → official BIRD EX).
# Args: <model_path|hf_id> <data_path> <tag> [gpu]
set -euo pipefail
source /home/datht/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate grpo 2>/dev/null || true
cd "$(dirname "$0")/.."

export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

MODEL="${1:?model path}"
DATA="${2:?data path}"
TAG="${3:?tag}"
GPU="${4:-0}"

EVAL_ROOT="output/final_eval"
OUT_DIR="${EVAL_ROOT}/${TAG}"
CACHE="${EVAL_ROOT}/exec_cache_${TAG}.pkl"
mkdir -p "$EVAL_ROOT" "$OUT_DIR"

# BIRD dev release location. Override via env var BIRD_DEV_ROOT.
# Default: ./data/bird/dev/ (relative to repo root)
BIRD_DEV_ROOT="${BIRD_DEV_ROOT:-data/bird/dev}"
DB_ROOT="${BIRD_DB_ROOT:-${BIRD_DEV_ROOT}/dev_databases/}"
GROUND_TRUTH="${BIRD_GOLD:-${BIRD_DEV_ROOT}/dev_gold.sql}"
DIFF_JSON="${BIRD_DIFF:-${BIRD_DEV_ROOT}/dev.json}"

if [[ ! -d "$DB_ROOT" || ! -f "$GROUND_TRUTH" || ! -f "$DIFF_JSON" ]]; then
    echo "ERROR: BIRD dev release not found." >&2
    echo "  expected DB_ROOT=$DB_ROOT" >&2
    echo "  expected GROUND_TRUTH=$GROUND_TRUTH" >&2
    echo "  expected DIFF_JSON=$DIFF_JSON" >&2
    echo "  Download BIRD dev from https://bird-bench.github.io/ and place at \$BIRD_DEV_ROOT (default ./data/bird/dev/)" >&2
    exit 1
fi

echo "=== [$(date)] Final 3B BIRD eval: $TAG ==="
echo "  model : $MODEL"
echo "  data  : $DATA"
echo "  out   : $OUT_DIR"
echo "  gpu   : $GPU"

CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_V1=0 python -u evaluation/evaluate_bird_dev.py \
    --model-path "$MODEL" --data-path "$DATA" \
    -n 30 --temperature 1.0 --batch-size 8 --max-samples -1 \
    --gpu-memory-utilization 0.85 \
    --exec-cache-file "$CACHE" --output-dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}/eval.log"

python -u evaluation/majority_voting.py \
    --input-pkl "${OUT_DIR}/detailed_results.pkl" \
    --output-file "${OUT_DIR}/mv_results.json" \
    --selection vav 2>&1 | tee "${OUT_DIR}/mv.log"

python -u evaluation/official_bird_evaluation/evaluation_bird_ex.py \
    --db_root_path "$DB_ROOT" --predicted_sql_json_path "${OUT_DIR}/predict_dev.json" \
    --data_mode dev --ground_truth_sql_path "$GROUND_TRUTH" --num_cpus 12 \
    --mode_predict gpt --diff_json_path "$DIFF_JSON" --meta_time_out 30.0 \
    2>&1 | tee "${OUT_DIR}/official_eval.log"

MV_ACC=$(python -c "import json; d=json.load(open('${OUT_DIR}/mv_results.json')); print(round(d.get('summary',d).get('accuracy',0)*100,2))")
RECALL=$(python -c "import json; d=json.load(open('${OUT_DIR}/metrics.json')); print(round(d.get('execution_recall_at_n',0)*100,2))" 2>/dev/null || echo "N/A")
# Parse the official BIRD evaluator table line: "accuracy <simple> <moderate> <challenging> <total>"
OFFICIAL_ACC=$(python3 -c "
import re
log = open('${OUT_DIR}/official_eval.log').read()
m = re.search(r'accuracy\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', log)
print(m.group(4) if m else 'N/A')
" 2>/dev/null || echo "N/A")

echo ""
echo "=== FINAL 3B BIRD ${TAG} ==="
echo "  Recall@30 : ${RECALL}%"
echo "  MV (vav)  : ${MV_ACC}%"
echo "  Official  : ${OFFICIAL_ACC}%"
echo "${TAG} mv=${MV_ACC} recall@30=${RECALL} official=${OFFICIAL_ACC}" >> "${EVAL_ROOT}/summary.txt"
