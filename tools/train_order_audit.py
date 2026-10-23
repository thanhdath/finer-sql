#!/usr/bin/env python3
"""Offline audit of the GRPO training-prompt order (TRL 0.25.1 RepeatSampler, single process).

Reproduces, without a GPU or a model, the prompt order grpo_writer.py's trainer draws, and prints the
same fingerprint grpo_writer.py writes to <output_dir>/train_order.json.

TRL 0.25.1 facts this relies on (trl/trainer/grpo_trainer.py `_get_train_sampler`, trl/trainer/utils.py
`RepeatSampler`, trl/trainer/grpo_config.py `__post_init__`):
  steps_per_generation  = gradient_accumulation_steps (when neither it nor generation_batch_size is set)
  generation_batch_size = per_device_train_batch_size * num_processes * steps_per_generation
  RepeatSampler(mini_repeat_count=num_generations, batch_size=generation_batch_size // num_generations,
                repeat_count=num_iterations * steps_per_generation, shuffle=True, seed=args.seed)
  The order is torch.randperm(n_rows, generator=torch.Generator().manual_seed(seed)), re-drawn from the same
  generator at every epoch. It depends ONLY on (n_rows, seed, prompts per generation batch), never on the
  row contents; two datasets therefore see the same prompt sequence iff, after load-time filtering, they
  have the same number of rows and the same (dataset_name, sample_id) at every position.
grpo_writer.py's own sampler override re-seeds with --data_seed when it is given and differs from --seed.

Load-time filtering: grpo_writer.py drops rows (GT-cache timeouts, optional --filter-empty-gt, prompt length
> --max_prompt_length, --limit_samples). Pass --gt_cache / --filter_empty_gt / --tokenizer + --max_prompt_length /
--limit_samples to replicate it; otherwise the audit ASSUMES NO ROWS ARE DROPPED and says so.

Usage:
  python tools/train_order_audit.py DATA [--seed 42] [--num_generations 32] [--per_device_train_batch_size 1]
                                         [--gradient_accumulation_steps 32] [--max_steps 3000]
  python tools/train_order_audit.py --compare A B [same options]   # identical order yes/no
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------- data loading (mirrors grpo_writer.py)
def load_rows(path: str) -> List[Dict[str, Any]]:
    if os.path.isfile(path) and path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    from datasets import load_from_disk
    ds = load_from_disk(path)
    if hasattr(ds, "keys") and callable(getattr(ds, "keys", None)):
        if "train" not in ds:
            raise SystemExit(f"{path}: DatasetDict without a 'train' split")
        ds = ds["train"]
    return ds.to_list()


def _first_gold(ex: Dict[str, Any]):
    g = ex["groundtruth_sqls"]
    g = list(g) if not isinstance(g, (str, list)) else g  # numpy arrays from parquet
    g0 = g[0]
    if not isinstance(g0, str) and not isinstance(g0, list):
        g0 = list(g0)
    return g0


def classify_gold(v: Any) -> str:  # copy of grpo_writer.classify_gold
    if v is None:
        return "gold_missing"
    if isinstance(v, dict):
        if v.get("timeout") is True:
            return "gold_timeout"
        if v.get("gold_error") is True:
            return "gold_error"
        rows = v.get("rows")
        return "gold_empty" if (not isinstance(rows, list) or len(rows) == 0) else "ok"
    if isinstance(v, list):
        return "gold_empty" if len(v) == 0 else "ok"
    return "gold_missing"


def apply_filters(rows: List[Dict[str, Any]], a) -> (List[Dict[str, Any]], List[str]):
    notes: List[str] = []
    if a.limit_samples and a.limit_samples > 0:
        rows = rows[: a.limit_samples]
        notes.append(f"limit_samples={a.limit_samples}")
    cache = None
    if a.gt_cache:
        import pickle
        with open(a.gt_cache, "rb") as f:
            cache = pickle.load(f)
    if cache is not None:
        timeouts = {k for k, v in cache.items() if classify_gold(v) == "gold_timeout"}
        if timeouts:
            n0 = len(rows)
            def keep_t(ex):
                g = _first_gold(ex)
                if not ex.get("dataset_name") or not ex.get("db_id") or not g:
                    return True
                return (ex["dataset_name"], ex["db_id"], g) not in timeouts
            rows = [ex for ex in rows if keep_t(ex)]
            notes.append(f"timeout filter dropped {n0 - len(rows)}")
        else:
            notes.append("timeout filter: cache has no timeout entries (dropped 0)")
        if a.filter_empty_gt:
            n0 = len(rows)
            def keep_e(ex):
                g = _first_gold(ex)
                if not ex.get("dataset_name") or not ex.get("db_id") or not g:
                    return True
                v = cache.get((ex["dataset_name"], ex["db_id"], g))
                if v is None or (isinstance(v, dict) and v.get("timeout") is True):
                    return True
                if isinstance(v, list):
                    return len(v) != 0
                r = v.get("rows") if isinstance(v, dict) else None
                return not (r is None or (isinstance(r, list) and len(r) == 0))
            rows = [ex for ex in rows if keep_e(ex)]
            notes.append(f"empty-GT filter dropped {n0 - len(rows)}")
    elif a.filter_empty_gt:
        notes.append("--filter_empty_gt given without --gt_cache: the trainer also skips it then")
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
        n0 = len(rows)
        def keep_l(ex):
            try:
                msgs = ex.get("messages")
                if msgs is not None and len(msgs) > 0:
                    msgs = [dict(m) for m in msgs]
                    text = tok.apply_chat_template(msgs, tokenize=False)
                else:
                    text = ex.get("prompt") or ex.get("prompt_raw") or ""
                g = _first_gold(ex)
                if isinstance(g, list):
                    g = g[0]
                return len(tok(text).input_ids) <= a.max_prompt_length and len(g) > 0
            except Exception:
                return False
        rows = [ex for ex in rows if keep_l(ex)]
        notes.append(f"length filter (<= {a.max_prompt_length} tokens) dropped {n0 - len(rows)}")
    if not a.gt_cache and not a.tokenizer:
        notes.append("ASSUMES NO ROWS DROPPED at load time (no --gt_cache/--tokenizer given)")
    return rows, notes


# ---------------------------------------------------------------- sampler + fingerprint
def get_repeat_sampler_cls():
    try:
        from trl.trainer.utils import RepeatSampler  # the class the trainer uses
        import trl
        return RepeatSampler, f"trl {trl.__version__}"
    except Exception:
        import torch

        class RepeatSampler:  # verbatim logic of trl 0.25.1 trl/trainer/utils.py
            def __init__(self, data_source, mini_repeat_count, batch_size=1, repeat_count=1, shuffle=True, seed=None):
                self.data_source, self.mini_repeat_count = data_source, mini_repeat_count
                self.batch_size, self.repeat_count = batch_size, repeat_count
                self.num_samples, self.shuffle, self.seed = len(data_source), shuffle, seed
                if shuffle:
                    self.generator = torch.Generator()
                    if seed is not None:
                        self.generator.manual_seed(seed)

            def __iter__(self):
                idx = (torch.randperm(self.num_samples, generator=self.generator).tolist()
                       if self.shuffle else list(range(self.num_samples)))
                chunks = [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
                for chunk in (c for c in chunks if len(c) == self.batch_size):
                    for _ in range(self.repeat_count):
                        for i in chunk:
                            for _ in range(self.mini_repeat_count):
                                yield i

            def __len__(self):
                return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count

        return RepeatSampler, "local copy of trl 0.25.1 RepeatSampler"


def fingerprint(rows: List[Dict[str, Any]], a) -> Dict[str, Any]:
    RS, impl = get_repeat_sampler_cls()
    G = a.num_generations
    spg = a.steps_per_generation or a.gradient_accumulation_steps
    gen_bs = a.per_device_train_batch_size * a.num_processes * spg
    if gen_bs % G:
        raise SystemExit(f"generation_batch_size {gen_bs} not divisible by num_generations {G}")
    B = gen_bs // G
    R = a.num_iterations * spg
    data_seed = a.data_seed
    sampler_seed = data_seed if (data_seed is not None and data_seed != a.seed) else a.seed
    sampler = RS(data_source=rows, mini_repeat_count=G, batch_size=B, repeat_count=R, shuffle=True, seed=sampler_seed)
    per_step = gen_bs * a.gradient_accumulation_steps
    need = per_step * a.max_steps
    epoch_len = len(sampler)
    if epoch_len == 0:
        raise SystemExit("empty sampler")
    # walk the flat stream once, epoch after epoch, exactly as the dataloader consumes it
    stream = hashlib.sha256()
    prompts: List[int] = []
    consumed, epochs = 0, 0
    block = G * B * R
    while consumed < need:
        epochs += 1
        for pos, idx in enumerate(sampler):
            if consumed >= need:
                break
            if pos % block < G * B and pos % G == 0:
                prompts.append(int(idx))
            stream.update(int(idx).to_bytes(8, "little", signed=True))
            consumed += 1
    has_ids = all(("dataset_name" in r and "sample_id" in r) for r in rows[:1])
    pair = lambda i: [rows[i].get("dataset_name"), _py(rows[i].get("sample_id"))]
    ids_hex = None
    if has_ids:
        h = hashlib.sha256()
        for i in prompts:
            h.update(json.dumps(pair(i)).encode() + b"\n")
        ids_hex = h.hexdigest()
    return {
        "seed": a.seed, "data_seed": data_seed, "sampler_seed": sampler_seed, "shuffle": True,
        "n_rows": len(rows), "num_generations": G, "prompts_per_generation_batch": B, "repeat_count": R,
        "per_device_train_batch_size": a.per_device_train_batch_size,
        "gradient_accumulation_steps": a.gradient_accumulation_steps, "steps_per_generation": spg,
        "generation_batch_size": gen_bs, "num_iterations": a.num_iterations, "max_steps": a.max_steps,
        "sampler_indices_per_step": per_step, "epoch_step_aligned": epoch_len % per_step == 0,
        "epochs_touched": epochs, "n_prompts": len(prompts),
        "first_64_indices": prompts[:64], "first_32_pairs": [pair(i) for i in prompts[:32]],
        "prompt_index_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "prompt_id_sha256": ids_hex, "sampler_stream_sha256": stream.hexdigest(),
        "_sampler_impl": impl,
    }


def _py(v):
    try:
        return v.item()  # numpy scalar -> python
    except AttributeError:
        return v


def audit(path: str, a) -> Dict[str, Any]:
    rows, notes = apply_filters(load_rows(path), a)
    fp = fingerprint(rows, a)
    fp["_path"] = path
    fp["_filter_notes"] = notes
    return fp


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", help="arrow dir (load_from_disk) or .parquet")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_seed", type=int, default=None)
    ap.add_argument("--num_generations", type=int, default=32)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=32)
    ap.add_argument("--steps_per_generation", type=int, default=None, help="default = gradient_accumulation_steps (TRL)")
    ap.add_argument("--num_iterations", type=int, default=1)
    ap.add_argument("--num_processes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--gt_cache", default=None, help="replicate the GT-cache timeout filter (one cache for both in --compare)")
    ap.add_argument("--gt_cache_b", default=None, help="--compare: cache for B if different from --gt_cache")
    ap.add_argument("--filter_empty_gt", action="store_true")
    ap.add_argument("--tokenizer", default=None, help="replicate the prompt-length filter with this tokenizer")
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args(argv)
    if a.num_processes != 1:
        print("WARNING: fingerprint models single-process consumption only", file=sys.stderr)
    show = lambda fp: {k: v for k, v in fp.items() if k not in ("first_64_indices", "first_32_pairs")}
    if a.compare:
        fa = audit(a.compare[0], a)
        if a.gt_cache_b:
            a.gt_cache = a.gt_cache_b
        fb = audit(a.compare[1], a)
        same_idx = fa["prompt_index_sha256"] == fb["prompt_index_sha256"]
        same_ids = fa["prompt_id_sha256"] is not None and fa["prompt_id_sha256"] == fb["prompt_id_sha256"]
        out = {"A": fa, "B": fb, "identical_index_order": same_idx, "identical_prompt_order": same_idx and same_ids}
        for tag, fp in (("A", fa), ("B", fb)):
            print(f"[{tag}] {json.dumps(show(fp))}")
            print(f"[{tag}] first 10 pairs: {fp['first_32_pairs'][:10]}")
        print(f"IDENTICAL ORDER: {'YES' if out['identical_prompt_order'] else 'NO'} "
              f"(index sha A={fa['prompt_index_sha256'][:16]} B={fb['prompt_index_sha256'][:16]}; "
              f"id sha A={str(fa['prompt_id_sha256'])[:16]} B={str(fb['prompt_id_sha256'])[:16]}; "
              f"n_prompts {fa['n_prompts']}/{fb['n_prompts']})")
        rc = 0 if out["identical_prompt_order"] else 1
    else:
        if not a.data:
            ap.error("give DATA or --compare A B")
        out = audit(a.data, a)
        print(json.dumps(show(out)))
        print(f"first 10 pairs: {out['first_32_pairs'][:10]}")
        rc = 0
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return rc


if __name__ == "__main__":
    sys.exit(main())
