# Review: --spurious_penalty + --pay_requires_ori_ok (reviewer claude-opus-5-5, 2026-10-24)

Reviewed commit 7f8863a (on 346c821, base 87e8af3). grpo_writer.py sha256 2065f9c9f6be1c219e999ec4be4d99c9e8958985a42384851d17d5cafb44be00.
Live APIs: CoRE :18221 (bundle proxy over BJ :18211), ORI :18231 (AB_original). Caches: CoRE f132bc32, ORI 5fd69b6b; op list of the
E12 CoRE parquet bab74990. Env: anaconda3/envs/grpo (TRL 0.25.1). Tests: review/test_review_spurious_penalty.py (+ reward_worker.py).

## Verdict: GO for CORE12_SP

| # | decisive test | result |
|---|---|---|
| 1 | verdict alignment under the 16-thread pool | PASS. Both futures are keyed by the rollout index (dicts `_futs`/`_ori_futs`), with no zip over completion order. Live: 8 groups x 32 mixed rollouts, every dumped `sql` matches its case and its own verdict. |
| 2 | four verdict combinations | PASS. Worker unit test (4 combos + --no_ and ORI-only variants). Live: paid on both -> 2.0/4.0 unchanged; wrong on both -> equal to flag off; spurious -> exec 0, atomic 0, executable 0, total = format + memory; CoRE-paid/ORI-mismatched -> 1 + atomic (unpaid memory path, unpaid for dyn mask). With the penalty on, --atomic_memo --memory_fast equals the default path (dump + Chroma ids). |
| 3 | ORI timeout never counts as ori_ok | PASS. Server-side timeout (mock) -> ori_ok False, spurious False, records equal to flag off. Dead ORI API (port 9, api_max_wait 3 s) -> no crash, same records as flag off. |
| 4 | flag off byte-identical to 87e8af3 | PASS. Reward path (default, --atomic_memo) on 96 live rollouts: rewards + dump records (minus ts) identical. Tiny CPU GRPO e2e (default; --atomic_memo --memory_fast): dump records + Chroma ids identical. |
| 5 | live audit SQL | PASS. 128/128 caught rollouts from caught_ori12_labelled.jsonl (behind population_ori12_caught.json): spurious true, exec 0, atomic 0. 93/93 unique CoRE-paid-but-ORI-mismatched SQLs from the CORE12 pod dump: demoted, none paid (exec < 2, mean exec 1.70). |

Overhead of the second verdict pass: median reward_fn time 0.079 s -> 0.111 s per 32 rollouts (+0.031 s), local, --no-memory.

Full suite (tests/ + review/): 50 passed, 6 failed. The 6 failures are all in tests/test_gold_guard_reward.py and come from the environment:
the default tokenizer path in that test is rejected as an HF repo id and :18601 is down. They fail the same way on 87e8af3.

## Notes (not blocking)
- Scope: --pay_requires_ori_ok is not in the original spec, which said "core_ok -> unchanged". It is ON by default with
  --spurious_penalty, so the staged launcher picks it up silently. RUN_SPEC_CORE12_SP_POD.json "why" describes only the spurious
  penalty, so it needs updating, or the launch should pass --no_pay_requires_ori_ok if the owner wants the original spec.
  Measured on CORE12 dump: 2.3 % of CoRE-paid rollouts are demoted. The inspected cases are genuinely wrong SQL (wrong aggregate, extra join),
  not ORI tie artefacts. The penalty is mild (exec 2.0 -> ~1.7).
- ORI API dead: each call waits up to --api_max_wait_time (300 s) and then falls back to the CoRE-only verdict. Nothing counts these
  fallbacks; spur/share dropping to 0 is the only sign. Suggest a spur/ori_err counter.
- An exception raised inside the ORI check_sql_correct crashes reward_fn. This is the same as the CoRE path and is not reachable
  in practice, because call_sql_api catches every exception.
- spurious is also set when the CoRE prediction failed to execute (e.g. a CoRE timeout) but ORI pays. The reward is the same as with the flag off (0),
  but spur/share is slightly inflated.
- Already present before this change: 1 + atomic == 2.0 counts as paid for the dyn mask and memory. No demoted rollout reached it (max atomic < 1).
- GT keys: CoRE and ORI caches have identical key sets; all 13,504 training golds are present and certifiable in both.
- Resume: only module-level lists, nothing in the checkpoint state. Eval calls cannot log metrics (guarded by model.training).
