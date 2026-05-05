#!/usr/bin/env bash
# Spider evaluation pipeline (vLLM n=30 → vav majority voting → official Spider EX).
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

echo "=== [$(date)] Final 3B Spider eval: $TAG ==="
echo "  model : $MODEL"
echo "  data  : $DATA"
echo "  out   : $OUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_V1=0 python -u evaluation/evaluate_bird_dev.py \
    --model-path "$MODEL" --data-path "$DATA" \
    -n 30 --temperature 1.0 --batch-size 8 --max-samples -1 \
    --gpu-memory-utilization 0.85 \
    --exec-cache-file "$CACHE" --output-dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}/eval.log"

python -u evaluation/majority_voting.py \
    --input-pkl "${OUT_DIR}/detailed_results.pkl" \
    --output-file "${OUT_DIR}/mv_results.json" \
    --selection vav 2>&1 | tee "${OUT_DIR}/mv.log"

# Spider official EX is computed by majority_voting.py via test_suite_sql_eval call.
MV_ACC=$(python -c "import json; d=json.load(open('${OUT_DIR}/mv_results.json')); print(round(d.get('summary',d).get('accuracy',0)*100,2))")
RECALL=$(python -c "import json; d=json.load(open('${OUT_DIR}/metrics.json')); print(round(d.get('execution_recall_at_n',0)*100,2))" 2>/dev/null || echo "N/A")
OFFICIAL_EX=$(python3 -c "
import re
log = open('${OUT_DIR}/mv.log').read()
m = re.search(r'execution\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', log)
print(round(float(m.group(5))*100, 2) if m else 'N/A')
" 2>/dev/null || echo "N/A")

echo ""
echo "=== FINAL 3B Spider ${TAG} ==="
echo "  Recall@30 : ${RECALL}%"
echo "  MV (vav)  : ${MV_ACC}%"
echo "  Official  : ${OFFICIAL_EX}%"
echo "${TAG} mv=${MV_ACC} recall@30=${RECALL} official=${OFFICIAL_EX}" >> "${EVAL_ROOT}/summary.txt"
