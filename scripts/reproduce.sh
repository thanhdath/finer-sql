#!/usr/bin/env bash
# One-shot reproduction of every EX number reported in the README,
# using scripts/eval_bird.sh and scripts/eval_spider.sh.
#
# Each run = vLLM n=30 @ temperature=1.0  →  vav majority voting  →  official EX.
# Run-to-run variance ≈ ±0.3 pp.
#
# Prereqs:
#   conda activate grpo
#   Prompt datasets are pulled automatically from HF (thanhdath/...).
#
# Usage (from repo root):
#   bash scripts/reproduce.sh                # run everything sequentially on GPU 0
#   GPU=1 bash scripts/reproduce.sh          # different GPU
#   bash scripts/reproduce.sh 3b_bird        # run a single tag
#
# Tags:
#   3b_bird     — FINER-SQL-3B-BIRD    on BIRD dev    → 67.80 % EX
#   0.5b_bird   — FINER-SQL-0.5B-BIRD  on BIRD dev    → 50.59 % EX
#   3b_spider   — FINER-SQL-3B-Spider  on Spider dev  → 85.0 % EX
#   0.5b_spider — FINER-SQL-0.5B-Spider on Spider dev → 75.0 % EX

set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-0}"
ONLY="${1:-all}"

source /home/datht/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate grpo 2>/dev/null || true

DATA_CACHE="data/reproduce"
mkdir -p "$DATA_CACHE"

export BIRD_DEV_ROOT="${BIRD_DEV_ROOT:-data/bird/dev}"
export SPIDER_DEV_ROOT="${SPIDER_DEV_ROOT:-data/spider}"

# Snapshot a HF dataset to local arrow format (the eval scripts need load_from_disk).
fetch_dataset () {
    local repo="$1" local_dir="$2"
    if [[ -f "${local_dir}/dataset_info.json" ]]; then
        return 0
    fi
    python -c "
from datasets import load_dataset
ds = load_dataset('${repo}', split='train')
ds.save_to_disk('${local_dir}')
"
}

run_3b_bird () {
    local data="${DATA_CACHE}/bird_dev_prompts"
    fetch_dataset thanhdath/bird_dev_prompts "$data"
    bash scripts/eval_bird.sh griffith-bigdata/FINER-SQL-3B-BIRD "$data" repro_3b_bird "$GPU"
}

run_0_5b_bird () {
    local data="${DATA_CACHE}/bird_dev_prompts"
    fetch_dataset thanhdath/bird_dev_prompts "$data"
    bash scripts/eval_bird.sh griffith-bigdata/FINER-SQL-0.5B-BIRD "$data" repro_0.5b_bird "$GPU"
}

run_3b_spider () {
    local data="${DATA_CACHE}/spider_dev_prompts"
    fetch_dataset thanhdath/spider_dev_prompts "$data"
    bash scripts/eval_spider.sh griffith-bigdata/FINER-SQL-3B-Spider "$data" repro_3b_spider "$GPU"
}

run_0_5b_spider () {
    local data="${DATA_CACHE}/spider_dev_prompts"
    fetch_dataset thanhdath/spider_dev_prompts "$data"
    bash scripts/eval_spider.sh griffith-bigdata/FINER-SQL-0.5B-Spider "$data" repro_0.5b_spider "$GPU"
}

case "$ONLY" in
    all)         run_3b_bird; run_0_5b_bird; run_3b_spider; run_0_5b_spider ;;
    3b_bird)     run_3b_bird ;;
    0.5b_bird)   run_0_5b_bird ;;
    3b_spider)   run_3b_spider ;;
    0.5b_spider) run_0_5b_spider ;;
    *) echo "Unknown tag: $ONLY"; exit 1 ;;
esac

echo
echo "==== summary ===="
grep "^repro_" output/final_eval/summary.txt 2>/dev/null || echo "(no runs yet)"
