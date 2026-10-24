"""--dapo / --dynamic_sampling mask / --lora_* in grpo_writer.py (CPU only, no GPU, no SQL API).

The grpo_writer functions and the APIAwareGRPOTrainer class are extracted from the source (AST) and run against a real
trl.GRPOTrainer with a tiny random Qwen2 on CPU, as in test_resume_order.py:
(a) dyn_group_keep: a group is kept iff 1 <= #paid <= G-1.
(b) mask: when every group is masked, advantages are 0, no Adam step happens (weights bit-identical, optimizer state
    empty), global_step still advances and dyn/skipped counts the steps; when groups have signal, the masked run is
    bit-for-bit the default run (same advantages, same final weights); in a mixed batch kept rows keep TRL's advantage.
(c) --dapo preset wiring and explicit-flag precedence; epsilon_high reaches the trainer.
(d) LoRA: LoraConfig from the flags; lora_merged_weights == peft merge_and_unload and never mutates the model;
    merged export loads as a plain HF model with identical logits; a LoRA GRPO run saves the adapter (+ merged/) and
    resumes from it.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_train_order_seed import TOKENIZER_DIR  # noqa: E402

FUNCS = ("_flag_given", "resolve_dapo_preset", "build_lora_config", "dyn_group_keep", "lora_merged_weights",
         "export_lora_merged")


def _ns(dyn_mode="off", merged_export=True):
    src = open(os.path.join(ROOT, "grpo_writer.py")).read()
    tree = ast.parse(src)
    seg: Dict[str, str] = {}
    cls = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            seg[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.ClassDef) and node.name == "APIAwareGRPOTrainer":
            cls = ast.get_source_segment(src, node)
    assert set(seg) == set(FUNCS) and cls
    from trl import GRPOTrainer
    ns: Dict[str, Any] = {"os": os, "sys": sys, "argparse": argparse, "Any": Any, "Dict": Dict, "List": List,
                          "Optional": Optional, "Tuple": Tuple, "GRPOTrainer": GRPOTrainer, "API_MAX_RETRIES": 3,
                          "wait_for_api_recovery": lambda url: False, "API_URL": None, "_DUMP": None,
                          "_LAST_EX": [], "DYN_MODE": dyn_mode,
                          "ARGS": argparse.Namespace(lora_no_merged_export=not merged_export)}
    for n in FUNCS:
        exec(seg[n], ns)
    exec(cls, ns)
    return ns


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def _model(tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM, set_seed
    set_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    return Qwen2ForCausalLM(cfg)


def _rows(n):
    return [{"prompt": f"question {i}?", "dataset_name": "bird", "sample_id": 1000 + i, "db_id": "db",
             "groundtruth_sqls": ["SELECT 1"]} for i in range(n)]


def _trainer(ns, out, paid_fn, max_steps=2, beta=0.0, ga=8, G=4, peft_config=None, save_steps=None, lr=1e-2,
             record=None):
    """paid_fn(sample_id, k) -> bool (rollout k of the group is paid). reward = 2.0*paid + a small k-dependent
    shaping term, so even unpaid groups have non-zero TRL advantages (as the atomic/memory terms do)."""
    from datasets import Dataset
    from trl import GRPOConfig
    tok = _tok()
    args = GRPOConfig(output_dir=out, use_cpu=True, bf16=False, fp16=False, report_to=[], per_device_train_batch_size=1,
                      gradient_accumulation_steps=ga, num_generations=G, max_steps=max_steps, beta=beta,
                      max_completion_length=4, max_prompt_length=32, seed=42, learning_rate=lr, logging_steps=1,
                      save_strategy=("steps" if save_steps else "no"), save_steps=(save_steps or 500),
                      num_train_epochs=-1, gradient_checkpointing=False)

    def reward(completions, sample_id=None, **k):
        ex, r = [], []
        for i, sid in enumerate(sample_id):
            p = bool(paid_fn(int(sid), i % G))
            ex.append(2.0 if p else 0.0)
            r.append((2.0 if p else 0.0) + 0.1 * ((i * 7) % 5))
        ns["_LAST_EX"].clear()
        ns["_LAST_EX"].extend(ex)
        return r

    tr = ns["APIAwareGRPOTrainer"](model=_model(tok), args=args, train_dataset=Dataset.from_list(_rows(8)),
                                   processing_class=tok, reward_funcs=reward,
                                   **({"peft_config": peft_config} if peft_config is not None else {}))
    if record is not None:
        orig = tr._generate_and_score_completions

        def wrapped(inputs):
            o = orig(inputs)
            record.append(o["advantages"].detach().clone())
            return o
        tr._generate_and_score_completions = wrapped
    return tr


def _params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ------------------------------------------------------------------ (a)
def test_dyn_group_keep():
    ns = _ns()
    k = ns["dyn_group_keep"]
    assert k([0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0], 4) == [False] * 4 + [True] * 4 + [False] * 4 + [True] * 4
    assert k([1.9, 0.0], 2) == [False, False]          # 'paid' means ex >= 2.0
    with pytest.raises(ValueError):
        k([0, 0, 0], 2)


# ------------------------------------------------------------------ (b)
@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_all_groups_masked_skips_every_optimizer_step(tmp_path):
    ns = _ns("mask")
    adv: List[torch.Tensor] = []
    tr = _trainer(ns, str(tmp_path), paid_fn=lambda sid, k: False, max_steps=2, record=adv)  # no paid rollout at all
    before = _params(tr.model)
    tr.train()
    assert tr.state.global_step == 2 and tr._dyn_skipped == 2 and tr._dyn_steps == 2
    assert all(bool((a == 0).all()) for a in adv) and len(adv) == 2
    after = _params(tr.model)
    assert all(torch.equal(before[n], after[n]) for n in before)
    assert len(tr.optimizer.optimizer.state if hasattr(tr.optimizer, "optimizer") else tr.optimizer.state) == 0
    logged = [h for h in tr.state.log_history if "dyn/skipped" in h]
    assert logged and logged[-1]["dyn/skipped"] == 2.0 and logged[-1]["dyn/step_skipped"] == 1.0


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_all_paid_group_is_masked_too(tmp_path):
    ns = _ns("mask")
    tr = _trainer(ns, str(tmp_path), paid_fn=lambda sid, k: True, max_steps=1)
    before = _params(tr.model)
    tr.train()
    assert tr._dyn_skipped == 1
    assert all(torch.equal(before[n], p) for n, p in _params(tr.model).items())


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_signal_groups_bit_identical_to_default(tmp_path):
    def paid(sid, k):
        return k in (0, 2)                      # every group 2/4 paid -> every group kept
    runs = {}
    for mode in ("off", "mask"):
        ns = _ns(mode)
        adv: List[torch.Tensor] = []
        tr = _trainer(ns, str(tmp_path / mode), paid_fn=paid, max_steps=2, beta=0.0, record=adv)  # tiny random model has no path for a ref copy
        tr.train()
        runs[mode] = (adv, _params(tr.model), tr)
    (a0, p0, _), (a1, p1, t1) = runs["off"], runs["mask"]
    assert t1._dyn_skipped == 0 and len(a0) == len(a1) == 2
    assert all(torch.equal(x, y) for x, y in zip(a0, a1))
    assert all(torch.equal(p0[n], p1[n]) for n in p0)
    assert any(not torch.equal(p0[n], _params(_model(_tok()))[n]) for n in p0)   # it really trained


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_mixed_batch_kept_rows_unchanged_masked_rows_zero(tmp_path):
    # 2 prompts per generation batch (G=4, GA=8): odd sample ids have signal, even ones are never paid
    def paid(sid, k):
        return sid % 2 == 1 and k == 1
    advs = {}
    for mode in ("off", "mask"):
        ns = _ns(mode)
        adv: List[torch.Tensor] = []
        tr = _trainer(ns, str(tmp_path / mode), paid_fn=paid, max_steps=1, beta=0.0, record=adv)
        sids: List[List[int]] = []
        orig = tr.reward_funcs[0]

        def rf(completions, sample_id=None, _o=orig, **k):
            sids.append([int(s) for s in sample_id])
            return _o(completions, sample_id=sample_id, **k)
        rf.__name__ = "reward"
        tr.reward_funcs = [rf]
        tr.train()
        advs[mode] = (adv[0], sids[0], tr)
    a0, s0, _ = advs["off"]
    a1, s1, t1 = advs["mask"]
    assert s0 == s1 and len({s % 2 for s in s0}) in (1, 2)
    for j, sid in enumerate(s1):
        if sid % 2 == 1:
            assert torch.equal(a1[j], a0[j])
        else:
            assert float(a1[j]) == 0.0
    assert any(float(a0[j]) != 0.0 for j, sid in enumerate(s0) if sid % 2 == 0) or all(s % 2 for s in s0)
    has_signal = any(s % 2 == 1 for s in s1)
    assert t1._dyn_skipped == (0 if has_signal else 1)


# ------------------------------------------------------------------ (c)
def _parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_type", default=None)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--epsilon_high", type=float, default=None)
    ap.add_argument("--dynamic_sampling", default="off")
    ap.add_argument("--dapo", action="store_true")
    return ap.parse_args(argv)


def test_dapo_preset_wiring_and_precedence():
    ns = _ns()
    r = ns["resolve_dapo_preset"]
    a = _parse([])
    assert r(a, []) == {"dapo": False, "loss_type": None, "epsilon_high": None, "beta": 0.04, "dynamic_sampling": "off"}
    a = _parse(["--dapo"])
    assert r(a, ["--dapo"]) == {"dapo": True, "loss_type": "dapo", "epsilon_high": 0.28, "beta": 0.0,
                                "dynamic_sampling": "mask"}
    argv = ["--dapo", "--beta", "0.01", "--loss_type=grpo", "--dynamic_sampling", "off", "--epsilon_high", "0.3"]
    a = _parse(argv)
    assert r(a, argv) == {"dapo": True, "loss_type": "grpo", "epsilon_high": 0.3, "beta": 0.01, "dynamic_sampling": "off"}


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_dapo_values_reach_trl(tmp_path):
    from datasets import Dataset
    from trl import GRPOConfig
    ns = _ns("mask")
    tok = _tok()
    cfg = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], per_device_train_batch_size=1,
                     gradient_accumulation_steps=4, num_generations=4, max_steps=1, loss_type="dapo", beta=0.0,
                     epsilon_high=0.28, max_completion_length=4, max_prompt_length=32, gradient_checkpointing=False)
    tr = ns["APIAwareGRPOTrainer"](model=_model(tok), args=cfg, train_dataset=Dataset.from_list(_rows(4)),
                                   processing_class=tok, reward_funcs=lambda completions, **k: [0.0] * len(completions))
    assert (tr.loss_type, tr.beta, tr.epsilon_low, tr.epsilon_high) == ("dapo", 0.0, 0.2, 0.28)
    assert tr.ref_model is None


# ------------------------------------------------------------------ (d)
def test_lora_config_from_flags():
    ns = _ns()
    b = ns["build_lora_config"]
    assert b(argparse.Namespace(lora_r=0)) is None
    c = b(argparse.Namespace(lora_r=64, lora_alpha=None, lora_dropout=0.05, lora_target="all-linear"))
    assert (c.r, c.lora_alpha, c.lora_dropout, c.target_modules, c.bias, str(c.task_type)) == \
        (64, 128, 0.05, "all-linear", "none", "CAUSAL_LM")
    c = b(argparse.Namespace(lora_r=8, lora_alpha=16, lora_dropout=0.0, lora_target="q_proj, v_proj"))
    assert set(c.target_modules) == {"q_proj", "v_proj"}


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_lora_merged_weights_equal_peft_merge_and_export(tmp_path):
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    ns = _ns()
    tok = _tok()
    pm = get_peft_model(_model(tok), ns["build_lora_config"](argparse.Namespace(lora_r=4, lora_alpha=8, lora_dropout=0.0,
                                                                                    lora_target="all-linear")))
    torch.manual_seed(1)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.5)                       # non-zero delta
    before = _params(pm)
    merged = dict(ns["lora_merged_weights"](pm))
    assert all(torch.equal(before[n], p) for n, p in _params(pm).items())   # out of place
    assert not any("lora" in n for n in merged)
    # reference merge on a copy (merge_and_unload mutates)
    import copy
    ref_model = copy.deepcopy(pm).merge_and_unload()
    ref = {n: p.detach() for n, p in ref_model.named_parameters()}
    assert set(ref) == set(merged)
    assert all(torch.allclose(ref[n], merged[n], atol=1e-6) for n in ref)
    assert any(not torch.equal(merged[n], before["base_model.model." + n.replace(".weight", ".base_layer.weight")])
               for n in merged if n.endswith("q_proj.weight"))
    d = ns["export_lora_merged"](pm, str(tmp_path / "merged"), tok)
    m2 = AutoModelForCausalLM.from_pretrained(d)
    x = torch.tensor([[1, 2, 3, 4]])
    pm.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(pm(x).logits, m2(x).logits, atol=1e-5)


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer dir absent")
def test_lora_grpo_saves_adapter_and_resumes(tmp_path):
    ns = _ns("mask")
    lc = ns["build_lora_config"](argparse.Namespace(lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_target="all-linear"))
    paid = lambda sid, k: k == 0  # noqa: E731
    tr = _trainer(ns, str(tmp_path), paid_fn=paid, max_steps=2, peft_config=lc, save_steps=2, lr=1e-2)
    tr.train()
    ck = tmp_path / "checkpoint-2"
    for f in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json",
              "merged/model.safetensors", "merged/config.json"):
        assert (ck / f).is_file(), f
    assert not (ck / "model.safetensors").exists()      # HF resume must take the adapter path
    trained = {n: p.detach().clone() for n, p in tr.model.named_parameters() if "lora_" in n}
    assert any(p.abs().sum() > 0 for n, p in trained.items() if "lora_B" in n)
    ns2 = _ns("mask")
    tr2 = _trainer(ns2, str(tmp_path), paid_fn=paid, max_steps=3, peft_config=lc, save_steps=None, lr=1e-2)
    seen = {}

    from transformers import TrainerCallback

    class CB(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **k):
            seen["gs"] = state.global_step
            seen["lora"] = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    tr2.add_callback(CB())
    tr2.train(resume_from_checkpoint=str(ck))
    assert seen["gs"] == 2 and tr2.state.global_step == 3
    assert all(torch.equal(trained[n], seen["lora"][n]) for n in trained)
