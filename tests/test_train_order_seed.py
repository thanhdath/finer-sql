"""Training-order determinism for grpo_writer.py (CPU only, no GPU, no SQL).

(a) same seed -> same prompt order; different seed / data_seed -> different order.
(b) the fingerprint grpo_writer.py writes (its own functions, extracted from the source file because importing
    grpo_writer runs its CLI/tokenizer/cache side effects) equals tools/train_order_audit.py's, and both equal the
    index stream the REAL TRL GRPOTrainer dataloader yields. The trainer is a real trl.GRPOTrainer built on CPU around
    a tiny randomly initialised Qwen2 model (no checkpoint loaded); the tokenizer is read from a local checkpoint dir
    (test skipped if absent).
"""
from __future__ import annotations

import ast
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_order_audit as toa  # noqa: E402

TOKENIZER_DIR = os.environ.get(
    "TRAIN_ORDER_TOKENIZER",
    "/home/datht/finer-sql-tkde/experiments/05_core05f_finer_config_20261023/ori_a6000/stage/finer05b_bird",
)


def _rows(n: int, gold_prefix: str = "SELECT 1") -> List[Dict[str, Any]]:
    return [{"prompt": f"question {i}?", "dataset_name": "bird" if i % 3 else "spider", "sample_id": 1000 + i,
             "db_id": f"db{i % 5}", "groundtruth_sqls": [f"{gold_prefix} -- {i}"]} for i in range(n)]


def _args(**kw):
    base = dict(seed=42, data_seed=None, num_generations=4, per_device_train_batch_size=1,
                gradient_accumulation_steps=4, steps_per_generation=None, num_iterations=1, num_processes=1,
                max_steps=30)
    base.update(kw)
    return type("A", (), base)()


# ------------------------------------------------------------------ (a) determinism
def test_same_seed_same_order_different_seed_different_order():
    rows = _rows(97)
    a1 = toa.fingerprint(rows, _args(seed=42))
    a2 = toa.fingerprint(rows, _args(seed=42))
    b = toa.fingerprint(rows, _args(seed=43))
    c = toa.fingerprint(rows, _args(seed=42, data_seed=7))
    assert a1["prompt_index_sha256"] == a2["prompt_index_sha256"]
    assert a1["first_64_indices"] == a2["first_64_indices"]
    assert a1["prompt_index_sha256"] != b["prompt_index_sha256"]
    assert a1["prompt_index_sha256"] != c["prompt_index_sha256"]
    # data_seed == seed is the same as no data_seed (TRL default path untouched)
    d = toa.fingerprint(rows, _args(seed=42, data_seed=42))
    assert d["prompt_index_sha256"] == a1["prompt_index_sha256"]


def test_order_ignores_row_contents_only_ids_matter():
    ori, core = _rows(97, "SELECT 1"), _rows(97, "SELECT 2")  # same ids/order, different references
    fa, fb = toa.fingerprint(ori, _args()), toa.fingerprint(core, _args())
    assert fa["prompt_id_sha256"] == fb["prompt_id_sha256"]
    swapped = list(core)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert toa.fingerprint(swapped, _args())["prompt_id_sha256"] != fa["prompt_id_sha256"]


def test_trl_sampler_equals_local_copy():
    import torch
    from trl.trainer.utils import RepeatSampler as TRL
    toa_local = toa.get_repeat_sampler_cls.__code__  # noqa: F841 (presence)
    rows = list(range(53))
    s1 = TRL(rows, mini_repeat_count=4, batch_size=2, repeat_count=3, shuffle=True, seed=5)
    # force the fallback implementation
    import builtins
    real_import = builtins.__import__

    def no_trl(name, *a, **k):
        if name.startswith("trl"):
            raise ImportError
        return real_import(name, *a, **k)

    builtins.__import__ = no_trl
    try:
        Local, impl = toa.get_repeat_sampler_cls()
    finally:
        builtins.__import__ = real_import
    assert impl.startswith("local copy")
    s2 = Local(rows, mini_repeat_count=4, batch_size=2, repeat_count=3, shuffle=True, seed=5)
    assert list(s1) + list(s1) == list(s2) + list(s2)  # two epochs from one generator
    assert len(s1) == len(s2)


# ------------------------------------------------------------------ (b) trainer path == audit tool
def _extract_from_grpo_writer():
    src_path = os.path.join(ROOT, "grpo_writer.py")
    src = open(src_path).read()
    tree = ast.parse(src)
    funcs, method = {}, None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("train_order_fingerprint", "write_train_order"):
            funcs[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.ClassDef) and node.name == "APIAwareGRPOTrainer":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "_get_train_sampler":
                    method = ast.get_source_segment(src, sub, padded=True)
    assert set(funcs) == {"train_order_fingerprint", "write_train_order"} and method
    from trl import GRPOTrainer
    ns: Dict[str, Any] = {"os": os, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
                          "GRPOTrainer": GRPOTrainer}
    exec(funcs["train_order_fingerprint"], ns)
    exec(funcs["write_train_order"], ns)
    exec("class SeededGRPOTrainer(GRPOTrainer):\n" + textwrap.indent(textwrap.dedent(method), "    "), ns)
    return ns


def _cpu_trainer(tmp_path, rows, seed, data_seed, G=4, ga=4, pdbs=1, max_steps=12):
    if not os.path.isdir(TOKENIZER_DIR):
        pytest.skip(f"tokenizer dir missing: {TOKENIZER_DIR}")
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from trl import GRPOConfig
    ns = _extract_from_grpo_writer()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg)
    args = GRPOConfig(output_dir=str(tmp_path / "out"), use_cpu=True, bf16=False, fp16=False, report_to=[],
                      per_device_train_batch_size=pdbs, gradient_accumulation_steps=ga, num_generations=G,
                      max_steps=max_steps, beta=0.0, max_completion_length=4, max_prompt_length=32,
                      seed=seed, data_seed=data_seed, logging_steps=1, save_strategy="no")
    trainer = ns["SeededGRPOTrainer"](model=model, args=args, train_dataset=Dataset.from_list(rows),
                                      processing_class=tok, reward_funcs=lambda completions, **k: [0.0] * len(completions))
    return ns, trainer


@pytest.mark.parametrize("seed,data_seed,pdbs,ga", [(42, None, 1, 4), (42, 7, 1, 4), (3, None, 2, 2)])
def test_trainer_fingerprint_equals_audit_and_real_dataloader(tmp_path, seed, data_seed, pdbs, ga):
    rows = _rows(23)
    G, max_steps = 4, 30  # 30 steps x 1 prompt > 23 rows -> crosses an epoch boundary
    ns, trainer = _cpu_trainer(tmp_path, rows, seed, data_seed, G=G, ga=ga, pdbs=pdbs, max_steps=max_steps)
    out_dir = str(tmp_path / "out")
    fp_trainer = ns["write_train_order"](trainer, out_dir, max_steps)
    assert fp_trainer is not None
    on_disk = json.load(open(os.path.join(out_dir, "train_order.json")))
    assert on_disk == json.loads(json.dumps(fp_trainer))
    fp_audit = toa.fingerprint(rows, _args(seed=seed, data_seed=data_seed, num_generations=G,
                                           per_device_train_batch_size=pdbs, gradient_accumulation_steps=ga,
                                           max_steps=max_steps))
    for k in ("prompt_index_sha256", "prompt_id_sha256", "sampler_stream_sha256", "first_64_indices",
              "first_32_pairs", "n_prompts", "n_rows", "sampler_seed", "generation_batch_size", "epochs_touched"):
        assert fp_trainer[k] == fp_audit[k], k
    assert fp_trainer["epochs_touched"] >= 2
    # the fingerprint did not advance the trainer's own sampler: the real dataloader (built afterwards, as in
    # trainer.train()) yields exactly the audited stream; walk max_steps x ga batches across epochs
    want = []
    s = toa.get_repeat_sampler_cls()[0](rows, mini_repeat_count=G, batch_size=(pdbs * ga) // G or 1,
                                         repeat_count=ga, shuffle=True,
                                         seed=data_seed if data_seed is not None else seed)
    dl = trainer.get_train_dataloader()
    got: List[int] = []
    need = max_steps * ga
    id_of = {(r["dataset_name"], r["sample_id"]): i for i, r in enumerate(rows)}
    while len(got) < need * pdbs * ga:
        for batch in dl:
            got.extend(id_of[(x["dataset_name"], x["sample_id"])] for x in batch)
            if len(got) >= need * pdbs * ga:
                break
    while len(want) < len(got):
        want.extend(list(s))
    assert got == want[: len(got)]


def test_data_seed_changes_trainer_order(tmp_path):
    rows = _rows(23)
    ns, t1 = _cpu_trainer(tmp_path / "a", rows, 42, None)
    _, t2 = _cpu_trainer(tmp_path / "b", rows, 42, 42)
    _, t3 = _cpu_trainer(tmp_path / "c", rows, 42, 9)
    f = lambda t: ns["train_order_fingerprint"](t._get_train_sampler(), t.args, t.train_dataset, 12)
    f1, f2, f3 = f(t1), f(t2), f(t3)
    assert f1["prompt_index_sha256"] == f2["prompt_index_sha256"]
    assert f1["prompt_index_sha256"] != f3["prompt_index_sha256"]
    assert f3["sampler_seed"] == 9 and f1["sampler_seed"] == 42
