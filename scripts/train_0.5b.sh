#!/usr/bin/env bash
# Joint GRPO training for 0.5B: continued from grpo_bird_0.5b/checkpoint-3000 (50.85% MV BIRD)
# on combined BIRD+Spider data. Mirrors the 3B joint recipe that took BIRD 66.88 → 67.80.
#
# Goal: push 0.5B past 51% Official EX on BIRD AND improve Spider zero-shot.
#
# Strategy:
#   - Start from output/grpo_bird_0.5b/checkpoint-3000 (already at 50.85% MV BIRD)
#   - Train on combined BIRD+Spider (20 049 samples) — same dataset used for 3B joint
#   - Lower learning rate (5e-6) to preserve BIRD knowledge while adapting to Spider
#   - num_generations 32 — matches the original 0.5B BIRD run that hit 50.85%
#   - Memory DISABLED (--no-memory): vLLM 0.10.2 cannot serve Qwen3-Embedding cleanly,
#     and the 3B joint run also disabled memory and still hit 67.80% — atomic + exec
#     reward alone gives a strong, stable signal.
#   - GPU 0: training + vLLM colocate (0.5B fits easily in 48 GB with rollouts)
#   - Save every 100 steps; max 1500 steps. Best is usually at 200-600.
#
# Usage:  bash scripts/train_0.5b.sh
# Logs:   output/grpo_joint_0.5b/training.log
#
# Configurable via env vars (sensible defaults below):
#   BASE_MODEL — starting checkpoint (HF id or local path)
#   DATA_ROOT  — joint BIRD+Spider training set (load_from_disk format)
#   GT_CACHE   — pickled execution cache for the GT SQLs
#   OUT_DIR    — where to write checkpoints and logs

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-griffith-bigdata/FINER-SQL-0.5B-BIRD}"
DATA_ROOT="${DATA_ROOT:-data/train/grpo_bird_spider_joint}"
GT_CACHE="${GT_CACHE:-data/cache/gt_rows_cache.pkl}"
GT_CACHE_FALLBACK="${GT_CACHE_FALLBACK:-data/cache/gt_rows_cache_combined.pkl}"
OUT_DIR="${OUT_DIR:-output/grpo_joint_0.5b}"

source /home/datht/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate grpo 2>/dev/null || true

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR/logs"

# ── 1) Networking hygiene FIRST — so all subsequent curls go direct ──────────
# The Griffith proxy on 192.168.1.x:8118 will silently swallow localhost calls
# unless we explicitly bypass it. Do this before any curl/health-check.
export VLLM_USE_V1=0
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — incompatible with
# vLLM CuMemAllocator (colocate mode) and triggers an assertion in cumem.py.
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export no_proxy="localhost,127.0.0.1,::1,192.168.1.0/24"
export NO_PROXY="$no_proxy"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

# ── 2) Sanity checks ─────────────────────────────────────────────────────────
[[ -d "$BASE_MODEL"   ]] || { echo "ERROR: base model missing: $BASE_MODEL"; exit 1; }
[[ -d "$DATA_ROOT"    ]] || { echo "ERROR: joint dataset missing: $DATA_ROOT"; exit 1; }
if [[ ! -f "$GT_CACHE" && -f "$GT_CACHE_FALLBACK" ]]; then GT_CACHE="$GT_CACHE_FALLBACK"; fi
[[ -f "$GT_CACHE" ]] || { echo "ERROR: gt_rows_cache.pkl missing"; exit 1; }

# Verify exec API: POST a real SELECT and check {"ok":true} (timeout 5s)
API_RESP=$(curl -s --max-time 5 -X POST http://localhost:8001/execute \
    -H 'content-type: application/json' \
    -d '{"sql":"SELECT 1 AS x","db_id":"california_schools","dataset_name":"bird","split":"dev","timeout_ms":2000}' || true)
if ! echo "$API_RESP" | grep -q '"ok":true'; then
    echo "ERROR: SQL exec API on http://localhost:8001 not returning ok=true"
    echo "       response: $API_RESP"
    echo "       check proxy (current http_proxy='${http_proxy:-}'), and fuser 8001/tcp"
    exit 1
fi
echo "  api check  : OK ($(echo "$API_RESP" | head -c 80)...)"
echo "  base model : $BASE_MODEL"
echo "  data       : $DATA_ROOT  ($(python3 -c 'from datasets import load_from_disk; print(len(load_from_disk("'"$DATA_ROOT"'"))) ' 2>/dev/null || echo '?') rows)"
echo "  gt_cache   : $GT_CACHE"
echo "  out_dir    : $OUT_DIR"

# ── 3) Kill stale processes (avoid GPU/socket clashes) ───────────────────────
echo "[$(date)] Killing stale grpo / vLLM / embedding processes..."
pkill -9 -u $(whoami) -f "grpo_writer.py"   2>/dev/null || true
pkill -9 -u $(whoami) -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u $(whoami) -f "VLLM::EngineCore" 2>/dev/null || true
fuser -k 9000/tcp 2>/dev/null || true
sleep 10

# ── 4) GRPO training on GPU 0 (colocate vLLM rollouts, no memory) ────────────
echo "[$(date)] Launching GRPO joint 0.5B on GPU 0 (no-memory mode)..."
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    --config_file configs/accelerate_single_gpu.yaml \
    grpo_writer.py \
    --model_path "$BASE_MODEL" \
    --data_root "$DATA_ROOT" \
    --gt_cache "$GT_CACHE" \
    --api_url http://localhost:8001/execute \
    --max_steps 1500 \
    --save_steps 100 \
    --save_total_limit 8 \
    --learning_rate 5e-6 \
    --num_generations 32 \
    --gradient_accumulation_steps 32 \
    --max_completion_length 2048 \
    --max_prompt_length 2048 \
    --use_vllm \
    --vllm_mode colocate \
    --use_liger_kernel \
    --no-memory \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/training.log"

echo "[$(date)] Training finished."

echo ""
echo "=== Next steps ==="
echo "  # Evaluate every saved checkpoint on BIRD V2 prompts (n=30, vav, official EX)"
echo "  bash eval_joint_0.5b.sh"
echo ""
echo "  # Or evaluate a single checkpoint:"
echo "  bash eval_final_3b_bird.sh \\"
echo "      $OUT_DIR/checkpoint-XXXX \\"
echo "      /home/datht/grast-sql/end2end/data-dev/bird_dev_top30_prompts_v2 \\"
echo "      joint_0.5b_stepXXXX 0"
