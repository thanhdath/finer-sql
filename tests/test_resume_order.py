"""--resume_from_checkpoint / --memory_snapshot_init in grpo_writer.py (CPU only, no GPU, no SQL).

(a) A real trl.GRPOTrainer (tiny random Qwen2 on CPU, as in test_train_order_seed.py) trains 3 steps and saves a full
    checkpoint; a NEW trainer with a larger max_steps resumes from it through grpo_writer's own resume path
    (check_resume_checkpoint + ResumeScheduleCallback + trainer.train(resume_from_checkpoint=<dir>)). The prompts it
    trains on must be exactly audit positions 3.. of the seeded order (tools/train_order_audit.py), global_step must be
    3 at train begin, the optimizer state must be the checkpoint's, and the LR must be the re-created linear schedule.
(b) seed_memory_store copies a snapshot (never opened) into an empty chroma_path and refuses a non-empty one.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_order_audit as toa  # noqa: E402
from test_train_order_seed import TOKENIZER_DIR, _args, _rows  # noqa: E402

NAMES = ("make_resume_schedule_callback", "check_resume_checkpoint", "seed_memory_store")


def _extract():
    src = open(os.path.join(ROOT, "grpo_writer.py")).read()
    tree = ast.parse(src)
    funcs, method = {}, None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            funcs[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.ClassDef) and node.name == "APIAwareGRPOTrainer":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "_get_train_sampler":
                    method = ast.get_source_segment(src, sub, padded=True)
    assert set(funcs) == set(NAMES) and method
    from trl import GRPOTrainer
    ns: Dict[str, Any] = {"os": os, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
                          "GRPOTrainer": GRPOTrainer, "SEED": 42}
    for n in NAMES:
        exec(funcs[n], ns)
    exec("class SeededGRPOTrainer(GRPOTrainer):\n" + textwrap.indent(textwrap.dedent(method), "    "), ns)
    return ns


def _trainer(ns, out, rows, max_steps, seen, lr=1e-3, G=4, ga=4):
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, set_seed
    from trl import GRPOConfig
    set_seed(0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    args = GRPOConfig(output_dir=out, use_cpu=True, bf16=False, fp16=False, report_to=[],
                      per_device_train_batch_size=1, gradient_accumulation_steps=ga, num_generations=G,
                      max_steps=max_steps, beta=0.0, max_completion_length=4, max_prompt_length=32, seed=42,
                      learning_rate=lr, logging_steps=1, save_strategy="steps", save_steps=3, num_train_epochs=-1)

    def reward(completions, sample_id=None, dataset_name=None, **k):
        seen.append(sorted(set(zip(dataset_name, sample_id))))
        return [float(i % 2) for i in range(len(completions))]  # non-zero advantages -> real updates

    return ns["SeededGRPOTrainer"](model=Qwen2ForCausalLM(cfg), args=args, train_dataset=Dataset.from_list(rows),
                                   processing_class=tok, reward_funcs=reward)


def test_resume_skips_exactly_global_step_prompts(tmp_path):
    if not os.path.isdir(TOKENIZER_DIR):
        pytest.skip(f"tokenizer dir missing: {TOKENIZER_DIR}")
    import torch
    from transformers import TrainerCallback
    ns = _extract()
    rows = _rows(23)
    order = toa.fingerprint(rows, _args(num_generations=4, gradient_accumulation_steps=4, max_steps=10))
    want = [[(rows[i]["dataset_name"], rows[i]["sample_id"])] for i in order["first_64_indices"][:10]]

    seen_a: List = []
    ta = _trainer(ns, str(tmp_path / "a"), rows, 6, seen_a)
    ta.train()
    assert seen_a == want[:6]  # uninterrupted run follows the audited order
    ckpt = str(tmp_path / "a" / "checkpoint-3")
    st = torch.load(os.path.join(ckpt, "optimizer.pt"), weights_only=False)["state"]

    seen_b: List = []
    tb = _trainer(ns, str(tmp_path / "b"), rows, 10, seen_b)
    gs = ns["check_resume_checkpoint"](ckpt, len(rows), tb.args.generation_batch_size, 4)
    assert gs == 3
    rec_path = str(tmp_path / "resume_record.json")
    cb = ns["make_resume_schedule_callback"](rec_path)
    probe: Dict[str, Any] = {}

    class Probe(TrainerCallback):
        def on_step_begin(self, args, state, control, optimizer=None, **kw):
            if "lr" not in probe:
                probe["lr"] = optimizer.param_groups[0]["lr"]
                probe["step"] = state.global_step
                p0 = optimizer.param_groups[0]["params"][0]
                probe["exp_avg_equal"] = torch.equal(optimizer.state[p0]["exp_avg"].cpu(), st[0]["exp_avg"].cpu())

    tb.add_callback(cb)
    tb.add_callback(Probe())
    tb.train(resume_from_checkpoint=ckpt)
    assert seen_b == want[3:10], (seen_b, want)  # resume step 4 (1-based) = audit position 4
    rec = json.load(open(rec_path))
    assert rec["global_step"] == 3 and rec["max_steps"] == 10 and rec["scheduler_last_epoch"] == 3
    assert rec["lr_set_at_resume"][0] == pytest.approx(1e-3 * (10 - 3) / 10)
    assert rec["lr_from_checkpoint_optimizer"][0] == pytest.approx(1e-3 * (6 - 3) / 6)  # old schedule's value
    assert probe["step"] == 3 and probe["lr"] == pytest.approx(1e-3 * 7 / 10) and probe["exp_avg_equal"]
    assert tb.state.global_step == 10


def test_resume_past_first_epoch_refused(tmp_path):
    ns = _extract()
    ck = tmp_path / "ck"
    ck.mkdir()
    for f in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (ck / f).write_bytes(b"x")
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 23}))
    with pytest.raises(SystemExit):
        ns["check_resume_checkpoint"](str(ck), 23, 32, 32)
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 22}))
    assert ns["check_resume_checkpoint"](str(ck), 23, 32, 32) == 22
    (ck / "rng_state.pth").unlink()
    with pytest.raises(SystemExit):
        ns["check_resume_checkpoint"](str(ck), 23, 32, 32)


def test_seed_memory_store_copies_and_refuses_nonempty(tmp_path):
    ns = _extract()
    snap = tmp_path / "snap"
    (snap / "seg").mkdir(parents=True)
    (snap / "chroma.sqlite3").write_bytes(b"sqlite-bytes")
    (snap / "chroma.sqlite3-wal").write_bytes(b"wal")
    (snap / "seg" / "data_level0.bin").write_bytes(b"\x00" * 10)
    before = {p: p.read_bytes() for p in snap.rglob("*") if p.is_file()}
    dst = tmp_path / "live" / "chroma_db"
    rec = ns["seed_memory_store"](str(snap), str(dst))
    assert (dst / "chroma.sqlite3").read_bytes() == b"sqlite-bytes"
    assert (dst / "seg" / "data_level0.bin").exists() and not (dst / "chroma.sqlite3-wal").exists()
    assert rec["n_files"] == 2 and rec["chroma_sqlite3_sha256_src"] == rec["chroma_sqlite3_sha256_dst"]
    assert {p: p.read_bytes() for p in snap.rglob("*") if p.is_file()} == before  # snapshot untouched
    with pytest.raises(SystemExit):
        ns["seed_memory_store"](str(snap), str(dst))  # non-empty target
    empty = tmp_path / "empty"
    empty.mkdir()
    ns["seed_memory_store"](str(snap), str(empty))
    assert (empty / "chroma.sqlite3").exists()
