#!/usr/bin/env bash
# Joint GRPO training: continued from FINER-SQL-3B-BIRD on combined BIRD+Spider data
# Goal: single model that reaches both 67.5% on BIRD AND 85% on Spider
#
# Strategy:
#   - Start from FINER-SQL-3B-BIRD (already at 67.54% BIRD, 83.2% Spider zero-shot)
#   - Train on combined BIRD train (9428) + Spider train (8659, with 3x GROUP BY aggregate-first oversample)
#   - Save every 100 steps for early-stop opportunity
#   - Lower learning rate (5e-6) to preserve BIRD knowledge
#   - Max 1000 steps; will stop training when a checkpoint passes both targets
#
# Usage: bash scripts/train_3b.sh
#
# Configurable via env vars (sensible defaults below):
#   BASE_MODEL — starting checkpoint (HF id or local path)
#   DATA_ROOT  — joint BIRD+Spider training set (load_from_disk format)
#   GT_CACHE   — pickled execution cache for the GT SQLs
#   OUT_DIR    — where to write checkpoints and logs

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-griffith-bigdata/FINER-SQL-3B-BIRD}"
DATA_ROOT="${DATA_ROOT:-data/train/grpo_bird_spider_joint}"
GT_CACHE="${GT_CACHE:-data/cache/gt_rows_cache_combined.pkl}"
OUT_DIR="${OUT_DIR:-output/grpo_joint_3b}"

echo "[$(date)] Killing stale processes..."
pkill -9 -f "grpo_writer.py" 2>/dev/null || true
pkill -9 -f "trl.*vllm-serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
fuser -k 51216/tcp 2>/dev/null || true

echo "Waiting 30s for GPU/sockets to release..."
sleep 30

source /home/datht/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate grpo 2>/dev/null || true

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR/logs"

# Reset chroma_db (memory module) for fresh start (joint training)
rm -rf ./memory/chroma_db-joint-3b
cp -r ./memory/chroma_db-init ./memory/chroma_db-joint-3b 2>/dev/null || mkdir -p ./memory/chroma_db-joint-3b

echo "[$(date)] Starting vLLM server on GPU 1 (port 8002)..."
CUDA_VISIBLE_DEVICES=1 \
    trl vllm-serve \
        --model "$BASE_MODEL" \
        --port 8002 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 \
        --dtype bfloat16 \
        --trust_remote_code \
    > "$OUT_DIR/logs/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM server..."
for i in $(seq 1 60); do
    curl -sf http://localhost:8002/health > /dev/null 2>&1 && { echo "vLLM ready!"; break; }
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died early"; tail -30 "$OUT_DIR/logs/vllm_server.log"; exit 1
    fi
    sleep 5
done

echo "[$(date)] Starting GRPO joint training on GPU 0..."
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    --config_file configs/accelerate_single_gpu.yaml \
    grpo_writer.py \
    --model_path "$BASE_MODEL" \
    --data_root "$DATA_ROOT" \
    --gt_cache "$GT_CACHE" \
    --api_url http://localhost:8001/execute \
    --max_steps 1000 \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 5e-6 \
    --num_generations 16 \
    --gradient_accumulation_steps 16 \
    --max_completion_length 2048 \
    --max_prompt_length 4096 \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host localhost \
    --vllm_server_port 8002 \
    --vllm-no-sleep \
    --no-memory \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/training.log"

echo "[$(date)] Training finished. Cleanup vLLM server..."
kill $VLLM_PID 2>/dev/null || true
sleep 5
