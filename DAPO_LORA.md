# Opt-in DAPO and LoRA for grpo_writer.py (reward-speed branch, 2026-10-24)

If none of the flags below is given, the trainer behaves exactly as it did at c778575. This was checked by running the
e2e worker on both versions: every dump field, including step and advantage, came out the same, and so did the
Chroma ids (`scratch/dapo_lora_logs/old_vs_new_default.py`).

## Flags
| flag | effect |
|---|---|
| `--dapo` | Preset: `loss_type dapo`, `epsilon_high 0.28` (epsilon low stays TRL's 0.2), `beta 0.0`, `dynamic_sampling mask`. A component given explicitly on the command line wins over the preset. The resolved values are printed (`[dapo_lora]`) and written to `<out_dir>/dapo_lora_recipe.json`. |
| `--loss_type`, `--epsilon_high`, `--beta` | Each one can also be set on its own. |
| `--dynamic_sampling mask` | A group is kept iff it has between 1 and G-1 paid rollouts (ex == 2.0). Masked groups get advantage 0 (`torch.where`, so kept rows are TRL's values bit for bit) and every rollout in the dump gets `dyn_mask`. If no micro-batch of an optimizer step holds a kept row, the Adam step is skipped (no momentum drift). global_step and the LR scheduler still advance, so the seed-42 order, the save points and resume do not change. Logged as `dyn/kept`, `dyn/masked`, `dyn/step_skipped`, `dyn/skipped` (cumulative) and a `[dyn] ... skipped` line. With beta 0, a fully masked micro-batch skips its forward and backward pass, because its gradient is exactly 0. |
| `--dynamic_sampling resample` | Deferred: it exits with a message. In TRL 0.25.1 each optimizer step draws exactly one generation batch from HF's epoch iterator, which lives inside `_inner_training_loop` and cannot be reached from `_prepare_inputs`. Drawing extra prompts would need a second sampler, and HF's resume skip (global_step x GA batches) would then replay the wrong prompts. That would break train_order.json and the resume audit. |
| `--lora_r R --lora_alpha A --lora_dropout D --lora_target all-linear` | peft `LoraConfig(task_type=CAUSAL_LM, bias=none)` is passed to `GRPOTrainer(peft_config=...)`. `lora_r 0` (the default) means full fine-tuning. |

## LoRA and vLLM weight sync
TRL 0.25.1 `GRPOTrainer._move_model_to_vllm` uses this path for PEFT models on one process without FSDP: it calls
`merge_adapter()`, then `load_weights` for each parameter (with names fixed up), then `unmerge_adapter()`. That merge
is done in place on the bf16 base weights, so every sync rounds them again and they drift. Our override uses
`lora_merged_weights()` instead: it computes base plus delta out of place in fp32 and casts once, so the trained model
is never mutated. FSDP, ZeRO-3 and server mode fall back to TRL's own function. Tests check the override against
peft `merge_and_unload`.

## Checkpoints
Each `checkpoint-N/` holds `adapter_model.safetensors` and `adapter_config.json`, plus the optimizer, scheduler, rng
state and trainer state. HF resume loads the adapter. `checkpoint-N/merged/` holds a plain HF model (`model.safetensors`,
config and tokenizer) for evaluation. It is not written to the top level, because HF resume would load a top-level
`model.safetensors` as full weights. Use `--lora_no_merged_export` to turn it off.

The HF backup script `hf_backup_arm.py` line 104 asserts `model.safetensors`, so it fails on LoRA checkpoints. To fix
it, accept `adapter_model.safetensors` (plus `merged/model.safetensors`) when `adapter_config.json` is present. Its
extension allowlist already covers both files.

## Spurious penalty (opt-in, 2026-10-24)
| flag | effect |
|---|---|
| `--spurious_penalty --ori_api_url URL --ori_gt_cache PATH` | CoRE arm only. Every rollout with an SQL is also judged on the original DB (same `check_sql_correct` comparator, submitted to the same 16-thread pool in the same pass). Paid on the arm's DB: unchanged (exec_score 2.0). Not paid on the arm's DB (with a usable gold there) but paid on the original DB: spurious, exec_score 0.0, atomic 0.0, executable 0.0; format and memory are computed as usual from exec_score 0 (unpaid memory path). Otherwise unchanged (1 + atomic if executable). The dyn mask sees a spurious rollout as unpaid. Dump adds `ori_ok` and `spurious`; logs `spur/share` (spurious / rollouts per generation batch) and `spur/groups` (groups with at least one spurious rollout). Both ORI flags are required when it is on. Tests: `tests/test_spurious_penalty.py`. |
| `--pay_requires_ori_ok` / `--no_pay_requires_ori_ok` | Default: on iff `--spurious_penalty`. A rollout is paid only if it is paid on the arm's DB AND on the original DB. Paid on the arm's DB but executed and mismatched against a usable gold on the original DB (e.g. the bundle's base member is the compact D0): wrong-executable, exec_score 1 + atomic, unpaid memory path, unpaid for the dyn mask. An original-DB execution error/timeout or unusable original gold keeps the arm's verdict. Dump adds `core_ok` (arm's verdict before this rule); log `spur/core_only` (count per generation batch). Needs the two ORI flags. |
