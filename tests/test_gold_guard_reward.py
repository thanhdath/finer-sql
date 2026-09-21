#!/usr/bin/env python3
"""A gold that cannot execute must never certify a prediction.

The defect this guards against
------------------------------
``build_gt_cache.py`` used to write ``cache[key] = []`` whenever the ground-truth SQL
raised -- the same value it writes for a gold that legitimately returns no row.  The
reward path then did::

    gt_rows = _GT_CACHE.get(key)                    # -> []
    ok = dataframes_equal(_rows_to_dataframe(gt_rows),
                          _rows_to_dataframe(pred_res.get("rows")))

and ``dataframes_equal`` compares row *sets*, so ``set() == set()`` -- **any** prediction
that executes and returns zero rows was a match, and collected the full execution reward
(``exec_score = 2.0``) plus, with memory on, an insertion of its reasoning trace.  In
other words ``SELECT <col> FROM <table> WHERE 1=0`` was a free maximum score on every
prompt whose gold is broken.

That is not hypothetical: 330 of the 373 BIRD ``retail_world`` golds in the training
corpus of record cannot execute, because BIRD's own train release ships a reduced
Northwind schema (8 tables, ``Customers.CustomerName``) while its golds are written
against the full one (``Order Details``, ``Territories``, ``EmployeeTerritories``,
``Customers.CompanyName``).  Corpus-wide, 360 golds error and a further 1,936 return
zero rows on the original-database endpoint.

These tests drive the REAL entry point: they import ``grpo_writer`` exactly as the
launcher does (module-level argparse, real GT-cache load, real ``check_sql_correct`` and
``reward_fn``) in a subprocess, and execute the predictions against the live
``db_execution`` API.  Nothing is stubbed.  They also grep the startup census that the
launcher prints, because a guard whose log line nobody can find is a guard nobody can
audit.

    python3 -m pytest tests/test_gold_guard_reward.py -q
    python3 tests/test_gold_guard_reward.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The API the campaign scores against (original BIRD/Spider databases).  Overridable so
# the test can be pointed at whichever db_execution endpoint is up.
API_URL = os.environ.get("GOLD_GUARD_API_URL", "http://127.0.0.1:18601/execute")
TOKENIZER = os.environ.get(
    "GOLD_GUARD_MODEL_PATH",
    "/home/datht/finer-sql-tkde/models/Qwen2.5-Coder-0.5B-Instruct",
)

# A real BIRD retail_world gold that does not execute on BIRD's own release.
BROKEN_GOLD = "SELECT CompanyName FROM Customers WHERE Country = 'Germany'"
# A real gold on the same database that does execute.
GOOD_GOLD = "SELECT CustomerName FROM Customers WHERE Country = 'Germany'"
# The free-lunch prediction: valid SQL, executes, returns zero rows.
EMPTY_PRED = "SELECT CustomerName FROM Customers WHERE 1=0"


def _api_up() -> bool:
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({"dataset_name": "bird", "db_id": "retail_world",
                             "sql": "SELECT 1 AS one", "mode": "read_only"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return bool(json.load(urllib.request.urlopen(req, timeout=10)).get("ok"))
    except Exception:
        return False


WORKER = r'''
import json, os, pickle, sys
sys.path.insert(0, os.environ["REPO"])
os.chdir(os.environ["REPO"])
cache_path = os.environ["CACHE"]
extra = json.loads(os.environ["EXTRA_ARGV"])
sys.argv = ["grpo_writer.py",
            "--model_path", os.environ["TOKENIZER"],
            "--gt_cache", cache_path,
            "--api_url", os.environ["API_URL"],
            "--no-memory", "--no-atomic"] + extra
import grpo_writer as G

BROKEN_GOLD = os.environ["BROKEN_GOLD"]
GOOD_GOLD = os.environ["GOOD_GOLD"]
EMPTY_PRED = os.environ["EMPTY_PRED"]
out = {}

# The comparator itself is unchanged -- it is the GATE in front of it that blocks the
# free lunch.  Record that, so a future refactor cannot "fix" this by breaking equality.
out["empty_vs_empty_still_equal"] = G.dataframes_equal(
    G._rows_to_dataframe([]), G._rows_to_dataframe([]))

v = G.check_sql_correct("bird", "retail_world", EMPTY_PRED, BROKEN_GOLD)
out["broken_gold_verdict_ok"] = v["ok"]
out["broken_gold_unusable"] = v.get("gold_unusable")
out["broken_gold_pred_executed"] = v["pred_res"].get("ok")

v = G.check_sql_correct("bird", "retail_world", EMPTY_PRED, "LEGACY_EMPTY")
out["legacy_empty_verdict_ok"] = v["ok"]
out["legacy_empty_unusable"] = v.get("gold_unusable")

v = G.check_sql_correct("bird", "retail_world", GOOD_GOLD, GOOD_GOLD)
out["good_gold_verdict_ok"] = v["ok"]
out["good_gold_unusable"] = v.get("gold_unusable")

def reward(pred, gold):
    # has_format_bonus requires >100 chars of thought, nothing before <think>, and
    # an extractable SQL answer -- otherwise the whole reward is gated to 0 and the
    # test would prove nothing about the execution component.
    thought = ("I read the schema, pick the table and the filter, and write one "
               "SQLite statement that answers the question exactly as asked. " * 3)
    comp = "<think>" + thought + "</think>" + pred
    return G.reward_fn(["prompt"], [comp],
                       groundtruth_sqls=[[gold]], db_id=["retail_world"],
                       dataset_name=["bird"])[0]

out["reward_broken_gold"] = reward(EMPTY_PRED, BROKEN_GOLD)
out["reward_legacy_empty"] = reward(EMPTY_PRED, "LEGACY_EMPTY")
out["reward_good_gold"] = reward(GOOD_GOLD, GOOD_GOLD)
out["gold_unusable_rollouts"] = G._gold_unusable_rollouts
out["gold_unusable_kinds"] = G._gold_unusable_kinds

print("###RESULT###" + json.dumps(out))
'''


def _execute(sql: str):
    """Run one statement through the db_execution API (never sqlite3 directly)."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"dataset_name": "bird", "db_id": "retail_world", "sql": sql,
                         "mode": "read_only", "timeout_ms": 120000}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=180))


_RUN_CACHE: dict = {}


def _run(extra_argv=None):
    """Import grpo_writer the way the launcher does and probe the real reward path."""
    memo = json.dumps(extra_argv or [])
    if memo in _RUN_CACHE:
        return _RUN_CACHE[memo]
    tmp = tempfile.mkdtemp(prefix="gold_guard_")
    cache_path = os.path.join(tmp, "gt.pkl")
    cache = {
        # post-fix marker written by build_gt_cache.py
        ("bird", "retail_world", BROKEN_GOLD): {
            "gold_error": True, "error": "no such column: CompanyName"},
        # what every cache built BEFORE the fix holds for a broken gold
        ("bird", "retail_world", "LEGACY_EMPTY"): [],
        # a real gold, cached as the rows the API really returns for it
        ("bird", "retail_world", GOOD_GOLD): _execute(GOOD_GOLD)["rows"],
    }
    assert cache[("bird", "retail_world", GOOD_GOLD)], \
        "the control gold must return at least one row, otherwise it proves nothing"
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    env = dict(os.environ)
    env.update(REPO=REPO, CACHE=cache_path, TOKENIZER=TOKENIZER, API_URL=API_URL,
               BROKEN_GOLD=BROKEN_GOLD, GOOD_GOLD=GOOD_GOLD, EMPTY_PRED=EMPTY_PRED,
               EXTRA_ARGV=json.dumps(extra_argv or []))
    proc = subprocess.run([sys.executable, "-c", WORKER], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=1800)
    if "###RESULT###" not in proc.stdout:
        raise AssertionError("worker produced no result\nSTDOUT:\n%s\nSTDERR:\n%s"
                             % (proc.stdout[-4000:], proc.stderr[-4000:]))
    payload = json.loads(proc.stdout.split("###RESULT###", 1)[1].splitlines()[0])
    _RUN_CACHE[memo] = (payload, proc.stdout)
    return _RUN_CACHE[memo]


def test_broken_gold_never_certifies_and_pays_zero():
    """The real defect: an errored gold + a zero-row prediction was a full-score match."""
    assert _api_up(), f"db_execution API not reachable at {API_URL}"
    r, _ = _run()

    # The prediction really did execute -- this is not a prediction-side failure.
    assert r["broken_gold_pred_executed"] is True
    # ...and the comparator would still call empty-vs-empty equal.  The gate is what stops it.
    assert r["empty_vs_empty_still_equal"] is True

    assert r["broken_gold_verdict_ok"] is False, "a gold that did not execute certified a prediction"
    assert r["broken_gold_unusable"] == "gold_error"
    # format bonus 1.0 only; the whole execution component is 0.
    assert r["reward_broken_gold"] == 1.0, r["reward_broken_gold"]


def test_legacy_empty_cache_entry_never_certifies():
    """Caches built before the fix store a broken gold as [] -- still never a match."""
    assert _api_up(), f"db_execution API not reachable at {API_URL}"
    r, _ = _run()
    assert r["legacy_empty_verdict_ok"] is False
    assert r["legacy_empty_unusable"] == "gold_empty"
    assert r["reward_legacy_empty"] == 1.0, r["reward_legacy_empty"]


def test_good_gold_is_unchanged():
    """A gold with real rows still scores exactly as before: exec 2.0 + format 1.0."""
    assert _api_up(), f"db_execution API not reachable at {API_URL}"
    r, _ = _run()
    assert r["good_gold_verdict_ok"] is True
    assert r["good_gold_unusable"] is None
    assert r["reward_good_gold"] == 3.0, r["reward_good_gold"]


def test_startup_census_is_printed_unconditionally():
    """The operator must be able to grep how many golds can never be scored."""
    assert _api_up(), f"db_execution API not reachable at {API_URL}"
    _, stdout = _run()
    assert "[gt_cache] gold_timeout=0 gold_error=1 gold_empty=1" in stdout, stdout[-3000:]
    assert "2 cached golds CANNOT certify a prediction" in stdout, stdout[-3000:]


def test_escape_hatch_restores_the_old_empty_behaviour():
    """--allow-empty-gt-match is the documented way to reproduce a pre-fix run.

    It restores the empty-gold match ONLY.  A gold explicitly recorded as an execution
    error stays unusable under every flag -- there is no way to buy that one back.
    """
    assert _api_up(), f"db_execution API not reachable at {API_URL}"
    r, _ = _run(["--allow-empty-gt-match"])
    assert r["legacy_empty_verdict_ok"] is True, "escape hatch did not restore old behaviour"
    assert r["broken_gold_verdict_ok"] is False, "gold_error must be unusable under every flag"
    assert r["reward_broken_gold"] == 1.0


def test_unusable_rollouts_are_counted_for_the_operator():
    """Every rollout scored against an unusable gold is counted and attributed by kind."""
    assert _api_up(), f"db_execution API not reachable at {API_URL}"
    r, _ = _run()
    assert r["gold_unusable_rollouts"] == 2, r["gold_unusable_rollouts"]
    assert r["gold_unusable_kinds"] == {"gold_error": 1, "gold_empty": 1}, r["gold_unusable_kinds"]


if __name__ == "__main__":
    if not _api_up():
        print(f"SKIP: db_execution API not reachable at {API_URL}")
        raise SystemExit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
