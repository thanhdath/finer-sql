![visitors](https://visitor-badge.laobi.icu/badge?page_id=thanhdath.finer-sql)

# FINER-SQL: Boosting Small Language Models for Text-to-SQL with Fine-Grained Execution Feedback and Cost-Efficient Rewards

**Models on Hugging Face:**
[FINER-SQL-3B-BIRD](https://huggingface.co/griffith-bigdata/FINER-SQL-3B-BIRD) ·
[FINER-SQL-3B-Spider](https://huggingface.co/griffith-bigdata/FINER-SQL-3B-Spider) ·
[FINER-SQL-0.5B-BIRD](https://huggingface.co/griffith-bigdata/FINER-SQL-0.5B-BIRD) ·
[FINER-SQL-0.5B-Spider](https://huggingface.co/griffith-bigdata/FINER-SQL-0.5B-Spider)

**Datasets on Hugging Face:**
[bird_dev_prompts](https://huggingface.co/datasets/griffith-bigdata/bird_dev_prompts) ·
[spider_dev_prompts](https://huggingface.co/datasets/griffith-bigdata/spider_dev_prompts)

#### Citation
```
@inproceedings{finersql,
  author       = {Thanh Dat Hoang and Thanh Trung Huynh and Matthias Weidlich and Thanh Tam Nguyen and Tong Chen and Hongzhi Yin and Quoc Viet Hung Nguyen},
  title        = {Boosting Small Language Models for Text-to-SQL with Fine-Grained Execution Feedback and Cost-Efficient Rewards},
  booktitle    = {ICDE},
  publisher    = {IEEE},
  year         = {2026},
}
```

---

FINER-SQL introduces **dense, interpretable rewards** to train **small language models (≤3B)** for Text-to-SQL via **Group Relative Policy Optimization (GRPO)**.  
Beyond from Format Reward and Execution Reward, it combines:
- **Memory Reward** — semantic alignment with verified reasoning traces  
- **Atomic Reward** — atomic operation-level SQL overlap for structural feedback

This helps solving the sparse reward issue of reinforcement learning in Text-to-SQL.

✅ Achieves 67.5% EX on BIRD when training only on BIRD train, and 85% EX on Spider using only a 3B model.  
⚡ Runs efficiently on a single 12-24 GB GPU.

-----------
**Backup Statistics**

![Visitors](https://margherita-gustatory-zane.ngrok-free.dev/badge/thanhdath%2Ffiner-sql.svg?ngrok-skip-browser-warning=true)

## Setup environment

```
conda create -n finer-sql python=3.10
conda activate finer-sql
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```


## Experiment Results

### BIRD Dev (n=1534)

| Model | Dataset (HF) | Recall@30 | MV (vav) | **Official EX** |
|---|---|---|---|---|
| **FINER-SQL-3B-BIRD** | [`bird_dev_prompts`](https://huggingface.co/datasets/griffith-bigdata/bird_dev_prompts) | 81.23 % | 67.67 % | **67.80 % ✅** |
| **FINER-SQL-0.5B-BIRD** | [`bird_dev_prompts`](https://huggingface.co/datasets/griffith-bigdata/bird_dev_prompts) | 68.32 % | 50.85 % | **50.59 % ✅** |

### Spider Dev (n=1034)

| Model | Dataset (HF) | Recall@30 | MV (vav) | **Official EX** |
|---|---|---|---|---|
| **FINER-SQL-3B-Spider** | [`spider_dev_prompts`](https://huggingface.co/datasets/griffith-bigdata/spider_dev_prompts) | 91.30 % | 85.88 % | **85.0 % ✅** |
| **FINER-SQL-0.5B-Spider** | [`spider_dev_prompts`](https://huggingface.co/datasets/griffith-bigdata/spider_dev_prompts) | 85.11 % | 75.44 % | **75.0 % ✅** |

### Difficulty breakdowns

**BIRD**

| Model | simple | moderate | challenging | total |
|---|---|---|---|---|
| FINER-SQL-3B-BIRD   | 73.51 % | 60.13 % | 55.86 % | 67.80 % |
| FINER-SQL-0.5B-BIRD | 60.11 % | 36.85 % | 33.79 % | 50.59 % |

**Spider**

| Model | easy | medium | hard | extra | total |
|---|---|---|---|---|---|
| FINER-SQL-3B-Spider   | 94.8 % | 90.1 % | 78.2 % | 64.5 % | 85.0 % |
| FINER-SQL-0.5B-Spider | 91.9 % | 82.5 % | 62.6 % | 42.8 % | 75.0 % |

---

## Reproduce numbers

```bash
git clone https://github.com/thanhdath/finer-sql.git && cd finer-sql
conda create -n grpo python=3.11 -y && conda activate grpo
pip install -r requirements.txt

# Prompt datasets are pulled automatically from HF (thanhdath/...).

# Run everything:
bash scripts/reproduce.sh

# Or a single tag:
GPU=0 bash scripts/reproduce.sh 3b_bird      # FINER-SQL-3B-BIRD on BIRD     → 67.80 % EX
GPU=0 bash scripts/reproduce.sh 0.5b_bird    # FINER-SQL-0.5B-BIRD on BIRD   → 50.59 % EX
GPU=0 bash scripts/reproduce.sh 3b_spider    # FINER-SQL-3B-Spider on Spider   → 85.0 % EX
GPU=0 bash scripts/reproduce.sh 0.5b_spider  # FINER-SQL-0.5B-Spider on Spider → 75.0 % EX
```

---

## Quick start (inference only)

```python
from vllm import LLM, SamplingParams

# Pick the size that fits your GPU; both work on BIRD and Spider.
llm = LLM(model="griffith-bigdata/FINER-SQL-3B-BIRD", dtype="bfloat16",
          max_model_len=4096, gpu_memory_utilization=0.85)
# Or: llm = LLM(model="griffith-bigdata/FINER-SQL-0.5B-BIRD", ...)

system = """You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.

Rules:
- Output exactly one SQL statement.
- The SQL must be executable on SQLite.
- Do not include any explanatory text.
- Output one SQL statement only. Do not include any extra text, tags, or code fences."""

sampling = SamplingParams(n=30, temperature=1.0, max_tokens=2048)
out = llm.chat([
    {"role": "system", "content": system},
    {"role": "user",   "content": f"Database Schema:\n{schema}\n\nQuestion: {question}"},
], sampling)
candidates = [c.text.split("</think>")[-1].strip() for c in out[0].outputs]
```

## Repository Structure

```
finer-sql/
├── README.md                  # this file
├── TRAIN_3B_BIRD_NO_GEN.md    # transferable training guide
├── grpo_writer.py             # GRPO training entry-point
├── sql_exec_scorer.py         # rank-correlation / footrule execution scorer
├── scripts/
│   ├── reproduce.sh           # one-shot result reproduction
│   ├── eval_bird.sh           # BIRD eval pipeline (n=30 vav + official EX)
│   ├── eval_spider.sh         # Spider eval pipeline
│   ├── train_3b.sh            # joint GRPO for the 3B family
│   └── train_0.5b.sh          # joint GRPO for the 0.5B family
├── configs/                   # accelerate / deepspeed configs
├── evaluation/                # n=30 generation + vav + official evaluators
├── atomic_ops/                # atomic-reward implementation
├── memory/                    # memory-reward (ChromaDB) infrastructure
├── db_execution/              # SQLite execution sandbox API (port 8001)
└── models/                    # local model cards
    ├── FINER-SQL-3B-BIRD/
    ├── FINER-SQL-3B-Spider/
    ├── FINER-SQL-0.5B-BIRD/
    └── FINER-SQL-0.5B-Spider/
```

---

## TODO

1. Merge Spider & BIRD specialists into a single unified model per size.
2. Train GRPO on SynSQL-2.5M with FINER-SQL's dense rewards.

---
