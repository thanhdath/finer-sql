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
    paid = sql == "PAID" or (ori and sql == "SPUR")
    return {"ok": paid, "pred_res": OK, "gt_res": OK}
G.check_sql_correct = fake
G.ATOMIC_OPS_REWARD.score_against_list = lambda p, g: 0.25
th = "<think>\n" + "x" * 120 + "\n</think>\n"
r = G.reward_fn(["p"] * 3, [th + "PAID", th + "SPUR", th + "WRONG"], groundtruth_sqls=[["G"]] * 3, db_id=["d"] * 3,
                dataset_name=["bird"] * 3, sample_id=[1] * 3)
G._DUMP.flush(0, None)
recs = [json.loads(l) for l in open(os.path.join(out, "r.jsonl"))]
print("###RESULT###" + json.dumps({"rewards": r, "recs": recs, "last_spur": G._LAST_SPUR}))
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


@pytest.mark.skipif(not os.path.isdir(TOKENIZER), reason="tokenizer dir absent")
def test_composition_three_cases():
    off = _unit([])
    on = _unit(["--spurious_penalty", "--ori_api_url", "http://ori/execute", "--ori_gt_cache", "@ORI"])
    assert off["rewards"] == [3.0, 2.25, 2.25]                 # format 1 + exec (2 | 1 + 0.25); memory off
    assert on["rewards"] == [3.0, 1.0, 2.25]                   # spurious: format only
    p, s, w = on["recs"]
    assert (s["exec_score"], s["atomic"], s["executable"], s["ex"], s["spurious"], s["ori_ok"]) == (0.0, 0.0, 0.0, 0.0, True, True)
    assert (p["spurious"], p["ori_ok"], p["exec_score"]) == (False, True, 2.0)
    assert (w["spurious"], w["ori_ok"], w["exec_score"], w["atomic"]) == (False, False, 1.25, 0.25)
    assert on["last_spur"] == [False, True, False]
    assert all("spurious" not in r and "ori_ok" not in r for r in off["recs"])
    drop = ("ts", "spurious", "ori_ok")
    assert [{k: v for k, v in r.items() if k not in drop} for r in (on["recs"][0], on["recs"][2])] == \
           [{k: v for k, v in r.items() if k not in drop} for r in (off["recs"][0], off["recs"][2])]


def _e2e(spurious):
    out = tempfile.mkdtemp(prefix="spur_e2e_")
    caches = {}
    for name, url in (("core", CORE_API), ("ori", ORI_API)):
        caches[name] = os.path.join(out, f"gt_{name}.pkl")
        pickle.dump({("bird", "mental_health_survey", GOLD): _exec(url, GOLD)["rows"]}, open(caches[name], "wb"))
    w = E.WORKER
    w = w.replace('SQLS = ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA", "SELECT COUNT(*) FROM CBSA WHERE CBSA > 0", "SELECT CBSA FROM CBSA LIMIT 1"]',
                  f'SQLS = [{GOLD!r}, {SPUR!r}, {WRONG!r}, {SPUR + ";"!r}]')
    w = w.replace('"db_id": "address"', '"db_id": "mental_health_survey"').replace(
        '"groundtruth_sqls": ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA"]', f'"groundtruth_sqls": [{GOLD!r}, {SPUR!r}]')
    assert SPUR in w and "mental_health_survey" in w
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
                assert 1.0 < a["exec_score"] < 2.0
    assert "spur/share" in log and "spur/groups" in log
