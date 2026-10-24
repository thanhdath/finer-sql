"""--spurious_penalty in grpo_writer.reward_fn.

(a) unit (no API): check_sql_correct is replaced by a fake keyed on api_url; three rollouts = paid on CoRE (4.0),
    spurious = paid only on the original DB (exec 0, atomic 0, total = format + memory), wrong on both (1 + atomic);
    flag off on the same verdicts gives the original values.
(b) e2e (live APIs, tiny CPU GRPO run as in test_reward_speed_e2e): CoRE bundle proxy :18221 as --api_url, original DB
    :18231 as --ori_api_url. Gold `SELECT COUNT(questiontext) FROM Question` (bird/mental_health_survey); the rollout
    `SELECT COUNT(*) FROM Question` matches it on the original DB (105 = 105) but not on the CoRE bundle (questiontext
    has NULLs in the CoRE members) -- taken from experiments/13_core_refusal_audit_20261024 caught_ori12_labelled.jsonl
    (sample 4618). Paid and wrong rollouts keep their values; the reward of the other rollouts equals the flag-off run.

    CORE_API_URL=http://127.0.0.1:18221/execute ORI_API_URL=http://127.0.0.1:18231/execute python -m pytest tests/test_spurious_penalty.py -q
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_reward_speed_e2e as E  # noqa: E402
from test_dump_rollouts import TOKENIZER  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_API = os.environ.get("CORE_API_URL", "http://127.0.0.1:18221/execute")
ORI_API = os.environ.get("ORI_API_URL", "http://127.0.0.1:18231/execute")
GOLD = "SELECT COUNT(questiontext) FROM Question"
SPUR = "SELECT COUNT(*) FROM Question"
WRONG = "SELECT COUNT(*) FROM Answer"
# CoRE-paid but ORI-wrong (symmetric_population.json, student_loan 4405): the bundle's base member is the compact D0, where
# one student has the max month; on the original DB 90 do.
CO_DB = "student_loan"
CO_GOLD = "SELECT name FROM longest_absense_from_school WHERE `month` = ( SELECT MAX(month) FROM longest_absense_from_school )"
CO_SQL = "SELECT DISTINCT longest_absense_from_school.name\nFROM longest_absense_from_school\nORDER BY longest_absense_from_school.month DESC\nLIMIT 1;"
CO_WRONG = "SELECT name FROM longest_absense_from_school WHERE `month` = 0"


def _exec(url, sql, db="mental_health_survey"):
    req = urllib.request.Request(url, data=json.dumps({"dataset_name": "bird", "db_id": db, "sql": sql, "mode": "read_only"}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def _up(url):
    try:
        return bool(_exec(url, "SELECT 1 AS one").get("ok"))
    except Exception:
        return False


UNIT = r'''
import json, os, pickle, sys
sys.path.insert(0, os.environ["REPO"]); os.chdir(os.environ["REPO"])
out = os.environ["OUT"]
sys.argv = ["grpo_writer.py", "--model_path", os.environ["TOKENIZER"], "--gt_cache", os.path.join(out, "gt.pkl"), "--api_url", "http://core/execute",
            "--no-memory", "--dump_rollouts", os.path.join(out, "r.jsonl"), "--out_dir", out] + json.loads(os.environ["EXTRA"])
import grpo_writer as G
OK = {"ok": True, "rows": [{"a": 1}]}
def fake(ds, db, sql, gt, api_url=None, gt_cache=None):
    ori = api_url is not None
    paid = sql == "PAID" or (sql == "SPUR" if ori else sql == "CORE_ONLY")
    return {"ok": paid, "pred_res": OK, "gt_res": OK}
G.check_sql_correct = fake
G.ATOMIC_OPS_REWARD.score_against_list = lambda p, g: 0.25
th = "<think>\n" + "x" * 120 + "\n</think>\n"
r = G.reward_fn(["p"] * 4, [th + "PAID", th + "SPUR", th + "WRONG", th + "CORE_ONLY"], groundtruth_sqls=[["G"]] * 4, db_id=["d"] * 4,
                dataset_name=["bird"] * 4, sample_id=[1] * 4)
G._DUMP.flush(0, None)
recs = [json.loads(l) for l in open(os.path.join(out, "r.jsonl"))]
print("###RESULT###" + json.dumps({"rewards": r, "recs": recs, "last_spur": G._LAST_SPUR, "last_core_only": G._LAST_CORE_ONLY}))
'''


def _unit(extra):
    out = tempfile.mkdtemp(prefix="spur_unit_")
    pickle.dump({}, open(os.path.join(out, "gt.pkl"), "wb"))
    pickle.dump({}, open(os.path.join(out, "ori.pkl"), "wb"))
    extra = [x.replace("@ORI", os.path.join(out, "ori.pkl")) for x in extra]
    env = dict(os.environ, REPO=REPO, OUT=out, TOKENIZER=TOKENIZER, CUDA_VISIBLE_DEVICES="", EXTRA=json.dumps(extra))
    r = subprocess.run([sys.executable, "-c", UNIT], env=env, capture_output=True, text=True, timeout=600)
    assert "###RESULT###" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(r.stdout.split("###RESULT###")[1].strip().splitlines()[0])


ORI = ["--ori_api_url", "http://ori/execute", "--ori_gt_cache", "@ORI"]


@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="tokenizer dir absent")
def test_composition_four_verdict_combinations():
    # rollouts: PAID (core ok, ori ok), SPUR (core no, ori ok), WRONG (core no, ori no), CORE_ONLY (core ok, ori no)
    off = _unit([])
    on = _unit(["--spurious_penalty"] + ORI)                   # implies --pay_requires_ori_ok
    assert off["rewards"] == [3.0, 2.25, 2.25, 3.0]            # format 1 + exec (2 | 1 + 0.25); memory off
    assert on["rewards"] == [3.0, 1.0, 2.25, 2.25]             # spurious: format only; core-only: wrong-executable
    assert _unit(["--spurious_penalty", "--no_pay_requires_ori_ok"] + ORI)["rewards"] == [3.0, 1.0, 2.25, 3.0]
    assert _unit(["--pay_requires_ori_ok"] + ORI)["rewards"] == [3.0, 2.25, 2.25, 2.25]
    p, s, w, c = on["recs"]
    assert (c["exec_score"], c["atomic"], c["executable"], c["ex"], c["verdict_ok"], c["core_ok"], c["ori_ok"], c["spurious"]) == \
           (1.25, 0.25, 1.0, 0.0, False, True, False, False)
    assert (p["core_ok"], s["core_ok"], w["core_ok"]) == (True, False, False)
    assert on["last_core_only"] == [False, False, False, True]
    on = dict(on, recs=on["recs"][:3], rewards=on["rewards"][:3]); off = dict(off, recs=off["recs"][:3])
    assert (s["exec_score"], s["atomic"], s["executable"], s["ex"], s["spurious"], s["ori_ok"]) == (0.0, 0.0, 0.0, 0.0, True, True)
    assert (p["spurious"], p["ori_ok"], p["exec_score"]) == (False, True, 2.0)
    assert (w["spurious"], w["ori_ok"], w["exec_score"], w["atomic"]) == (False, False, 1.25, 0.25)
    assert on["last_spur"] == [False, True, False, False]
    assert all("spurious" not in r and "ori_ok" not in r for r in off["recs"])
    drop = ("ts", "spurious", "ori_ok", "core_ok")
    assert [{k: v for k, v in r.items() if k not in drop} for r in (on["recs"][0], on["recs"][2])] == \
           [{k: v for k, v in r.items() if k not in drop} for r in (off["recs"][0], off["recs"][2])]


def _e2e(spurious, db="mental_health_survey", gold=GOLD, sqls=(GOLD, SPUR, WRONG, SPUR + ";")):
    out = tempfile.mkdtemp(prefix="spur_e2e_")
    caches = {}
    for name, url in (("core", CORE_API), ("ori", ORI_API)):
        caches[name] = os.path.join(out, f"gt_{name}.pkl")
        pickle.dump({("bird", db, gold): _exec(url, gold, db)["rows"]}, open(caches[name], "wb"))
    w = E.WORKER
    w = w.replace('SQLS = ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA", "SELECT COUNT(*) FROM CBSA WHERE CBSA > 0", "SELECT CBSA FROM CBSA LIMIT 1"]',
                  f'SQLS = {list(sqls)!r}')
    w = w.replace('"db_id": "address"', f'"db_id": {db!r}').replace(
        '"groundtruth_sqls": ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA"]', f'"groundtruth_sqls": [{gold!r}]')
    assert repr(sqls[1]) in w and repr(db) in w
    extra = ["--spurious_penalty", "--ori_api_url", ORI_API, "--ori_gt_cache", caches["ori"]] if spurious else []
    env = dict(os.environ, REPO=REPO, OUT=out, CACHE=caches["core"], API_URL=CORE_API, TOKENIZER=TOKENIZER, CUDA_VISIBLE_DEVICES="",
               EXTRA=json.dumps(extra))
    r = subprocess.run([sys.executable, "-c", w], env=env, capture_output=True, text=True, timeout=900)
    assert "###RESULT###" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout, [json.loads(l) for l in open(os.path.join(out, "rollouts.jsonl"))]


@pytest.mark.skipif(not (_up(CORE_API) and _up(ORI_API)), reason="CoRE/original db_execution APIs not reachable")
@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="tokenizer dir absent")
def test_e2e_live_apis():
    _, off = _e2e(False)
    log, on = _e2e(True)
    assert len(on) == len(off) == 16
    for a, b in zip(on, off):
        kind = {GOLD: "paid", WRONG: "wrong"}.get(a["sql"], "spur")
        assert a["sql"] == b["sql"]
        if kind == "spur":
            assert (b["ex"], b["executable"]) == (0.0, 1.0)            # flag off: CoRE refuses it but pays 1 + atomic
            assert (a["exec_score"], a["atomic"], a["executable"], a["ex"], a["spurious"], a["ori_ok"]) == (0.0, 0.0, 0.0, 0.0, True, True)
            assert a["total"] == a["format"] + a["memory"]
        else:
            assert a["spurious"] is False and a["ori_ok"] == (kind == "paid")
            for k in ("format", "ex", "executable", "atomic", "exec_score", "verdict_ok"):
                assert a[k] == b[k], (kind, k)
            if kind == "paid":
                assert a["total"] == b["total"] == 4.0
            else:
                assert a["executable"] == 1.0 and 1.0 <= a["exec_score"] < 2.0
    assert "spur/share" in log and "spur/groups" in log


@pytest.mark.skipif(not (_up(CORE_API) and _up(ORI_API)), reason="CoRE/original db_execution APIs not reachable")
@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="tokenizer dir absent")
def test_e2e_core_paid_ori_wrong_is_unpaid():
    sqls = (CO_GOLD, CO_SQL, CO_WRONG, CO_SQL)
    _, off = _e2e(False, CO_DB, CO_GOLD, sqls)
    log, on = _e2e(True, CO_DB, CO_GOLD, sqls)
    assert len(on) == len(off) == 16
    for a, b in zip(on, off):
        kind = {CO_GOLD: "paid", CO_WRONG: "wrong"}.get(a["sql"], "core_only")
        assert a["sql"] == b["sql"] and a["spurious"] is False
        if kind == "core_only":
            assert (b["ex"], b["exec_score"], b["total"]) == (2.0, 2.0, 4.0)      # flag off: CoRE pays it
            assert (a["core_ok"], a["ori_ok"], a["ex"], a["executable"], a["verdict_ok"]) == (True, False, 0.0, 1.0, False)
            assert 1.0 <= a["exec_score"] < 2.0 and a["exec_score"] == 1.0 + a["atomic"]
            assert a["total"] == a["format"] + a["exec_score"] + a["memory"]
        else:
            assert (a["core_ok"], a["ori_ok"]) == ((True, True) if kind == "paid" else (False, False))
            for k in ("format", "ex", "executable", "atomic", "exec_score", "verdict_ok"):
                assert a[k] == b[k], (kind, k)
    assert "spur/core_only" in log
