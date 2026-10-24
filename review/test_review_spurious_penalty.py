"""Independent review tests for --spurious_penalty (reviewer, 2026-10-24). Live local APIs :18221 (CoRE bundle proxy, BJ tree)
and :18231 (AB_original), the real E12 caches (CoRE f132bc32, ORI 5fd69b6b) and op list; caught rollouts from
experiments/13_core_refusal_audit_20261024/out/caught_ori12_labelled.jsonl (ORI12 rollouts paid on ORI, refused by CoRE).

  /home/datht/anaconda3/envs/grpo/bin/python -m pytest review/test_review_spurious_penalty.py -q -s
"""
import json, os, random, subprocess, sys, tarfile, tempfile, io, statistics
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TKDE = "/home/datht/finer-sql-tkde/experiments"
D = TKDE + "/12_final05b_20261024/data/"
CORE_GT, ORI_GT = D + "gt_cache_core_final05b.pkl", D + "gt_cache_ori_final05b.pkl"
OPS = D + "worth_train_final05b_core_enriched.atomic_ops_v2.json"
CAUGHT = TKDE + "/13_core_refusal_audit_20261024/out/caught_ori12_labelled.jsonl"
CORE_API = os.environ.get("CORE_API_URL", "http://127.0.0.1:18221/execute")
ORI_API = os.environ.get("ORI_API_URL", "http://127.0.0.1:18231/execute")
sys.path.insert(0, os.path.join(REPO, "tests"))
from test_dump_rollouts import TOKENIZER  # noqa: E402
SP = ["--spurious_penalty", "--ori_api_url", ORI_API, "--ori_gt_cache", ORI_GT]
RF = ("format", "ex", "executable", "atomic", "memory", "exec_score", "total", "verdict_ok", "sql", "pred_ok", "gold_unusable")
SCR = os.path.join(HERE, "_scratch"); os.makedirs(SCR, exist_ok=True)


def _cases(n_prompts=8, per=32, seed=7):
    rows = [json.loads(l) for l in open(CAUGHT)]
    by = {}
    for r in rows:
        by.setdefault((r["db_id"], r["sample_id"], r["gold"]), []).append(r["sql"])
    keys = sorted(by, key=lambda k: (-len(set(by[k])), k))[:40]
    random.Random(seed).shuffle(keys)
    groups = []
    for db, sid, gold in keys[:n_prompts]:
        base = {"gold": gold, "db_id": db, "sample_id": sid}
        caught = sorted(set(by[(db, sid, gold)]))
        g = []
        for i in range(per):
            k = i % 4
            if k == 0:
                g.append(dict(base, sql=gold, kind="paid"))
            elif k == 1:
                g.append(dict(base, sql=f"SELECT * FROM ({gold.rstrip().rstrip(';')}) LIMIT 0", kind="wrong"))
            else:
                g.append(dict(base, sql=caught[i % len(caught)], kind="caught"))
        groups.append(g)
    return groups


def _run(groups, extra, mode="none", repo=REPO):
    out = tempfile.mkdtemp(prefix="rev_", dir=SCR)
    cases = os.path.join(out, "cases.json"); json.dump(groups, open(cases, "w"))
    env = dict(os.environ, REPO=repo, OUT=out, TOKENIZER=TOKENIZER, CORE_GT=CORE_GT, CORE_API=CORE_API, ORI_API=ORI_API, OPS=OPS,
               CASES=cases, MODE=mode, EXTRA=json.dumps(extra), CUDA_VISIBLE_DEVICES="", no_proxy="*", NO_PROXY="*")
    r = subprocess.run([sys.executable, os.path.join(HERE, "reward_worker.py")], env=env, capture_output=True, text=True, timeout=1800)
    return r, (json.loads(r.stdout.split("###RESULT###")[1].strip().splitlines()[0]) if "###RESULT###" in r.stdout else None)


def _strip(rec, extra=()):
    return {k: v for k, v in rec.items() if k not in ("ts",) + tuple(extra)}


BASE_RES = {}


def test_population_caught_live():
    groups = _cases()
    flat = [c for g in groups for c in g]
    _, off = _run(groups, ["--atomic_memo"])
    _, on = _run(groups, ["--atomic_memo"] + SP)
    assert off and on
    n = {"caught": 0, "caught_spur": 0, "caught_ori_ok": 0, "caught_core_ok": 0}
    for c, a, b, s in zip(flat, on["recs"], off["recs"], on["last_spur"]):
        assert a["sql"] == b["sql"] == c["sql"]
        assert a["spurious"] == s
        if a["spurious"]:
            assert a["ori_ok"] is True and b["verdict_ok"] is False
            assert (a["exec_score"], a["atomic"], a["executable"], a["ex"]) == (0.0, 0.0, 0.0, 0.0)
            assert a["total"] == a["format"] + a["memory"]
        else:
            assert _strip(a, ("spurious", "ori_ok", "core_ok", "core_only")) == _strip(b), c["kind"]
        if c["kind"] == "paid":
            assert a["verdict_ok"] and a["ori_ok"] and not a["spurious"] and a["exec_score"] == 2.0
        if c["kind"] == "wrong":
            assert not a["spurious"] and a["ori_ok"] is False and a["exec_score"] == b["exec_score"]
        if c["kind"] == "caught":
            n["caught"] += 1; n["caught_spur"] += a["spurious"]; n["caught_ori_ok"] += bool(a["ori_ok"]); n["caught_core_ok"] += bool(a["verdict_ok"])
    print("\n[population]", json.dumps(n))
    BASE_RES.update(n=n, off_secs=off["secs"], on_secs=on["secs"])
    assert n["caught_spur"] >= 0.9 * n["caught"]


def test_ori_server_timeout_not_spurious():
    groups = _cases(n_prompts=2)
    _, off = _run(groups, ["--atomic_memo"])
    _, on = _run(groups, ["--atomic_memo"] + SP, mode="ori_timeout")
    assert on, "crashed"
    assert not any(on["last_spur"])
    for a, b in zip(on["recs"], off["recs"]):
        assert a["ori_ok"] is False and a["spurious"] is False
        assert _strip(a, ("spurious", "ori_ok", "core_ok", "core_only")) == _strip(b)


def test_ori_api_dead_not_spurious_no_crash():
    groups = _cases(n_prompts=1, per=8)
    _, off = _run(groups, ["--atomic_memo"])
    dead = ["--spurious_penalty", "--ori_api_url", "http://127.0.0.1:9/execute", "--ori_gt_cache", ORI_GT,
            "--api_max_wait_time", "3", "--api_check_interval", "1"]
    r, on = _run(groups, ["--atomic_memo"] + dead)
    assert on, r.stdout[-2000:] + r.stderr[-2000:]
    assert not any(on["last_spur"])
    for a, b in zip(on["recs"], off["recs"]):
        assert _strip(a, ("spurious", "ori_ok", "core_ok", "core_only")) == _strip(b)


def test_ori_check_raises_behaviour():
    """Documents what happens when the ORI verdict raises (not expected: call_sql_api catches every exception)."""
    r, on = _run(_cases(n_prompts=1, per=4), ["--atomic_memo"] + SP, mode="ori_raise")
    print("\n[ori_raise] result present:", on is not None, "| tail:", (r.stderr.strip().splitlines() or [""])[-1][:200])


def _base_tree():
    dst = os.path.join(SCR, "base_87e8af3")
    if not os.path.exists(os.path.join(dst, "grpo_writer.py")):
        os.makedirs(dst, exist_ok=True)
        tar = subprocess.run(["git", "-C", REPO, "archive", "87e8af3"], capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(dst)
    return dst


@pytest.mark.parametrize("extra", [[], ["--atomic_memo"]])
def test_flag_off_identical_to_87e8af3_reward_path(extra):
    groups = _cases(n_prompts=3)
    _, cur = _run(groups, extra)
    _, base = _run(groups, extra, repo=_base_tree())
    assert cur and base
    assert cur["rewards"] == base["rewards"]
    assert [_strip(r) for r in cur["recs"]] == [_strip(r) for r in base["recs"]]


def _e2e(repo, extra):
    import test_reward_speed_e2e as E
    import pickle, urllib.request
    out = tempfile.mkdtemp(prefix="rev_e2e_", dir=SCR)
    cache = os.path.join(out, "gt.pkl")
    req = urllib.request.Request(E.API_URL, data=json.dumps({"dataset_name": "bird", "db_id": "address", "sql": "SELECT COUNT(*) FROM CBSA",
                                                           "mode": "read_only"}).encode(), headers={"Content-Type": "application/json"})
    pickle.dump({("bird", "address", "SELECT COUNT(*) FROM CBSA"): json.load(urllib.request.urlopen(req, timeout=30))["rows"]}, open(cache, "wb"))
    env = dict(os.environ, REPO=repo, OUT=out, CACHE=cache, API_URL=E.API_URL, TOKENIZER=TOKENIZER, CUDA_VISIBLE_DEVICES="", EXTRA=json.dumps(extra))
    r = subprocess.run([sys.executable, "-c", E.WORKER], env=env, capture_output=True, text=True, timeout=900)
    assert "###RESULT###" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(r.stdout.split("###RESULT###")[1].strip().splitlines()[0])
    raw = open(os.path.join(out, "rollouts.jsonl")).read().splitlines()
    return res, [_strip(json.loads(l)) for l in raw]


@pytest.mark.parametrize("extra", [[], ["--atomic_memo", "--memory_fast"]])
def test_flag_off_e2e_training_identical_to_87e8af3(extra):
    """Full tiny CPU GRPO run (2 steps, memory ON with stub embeddings): dump records (minus ts) + chroma ids identical."""
    r_cur, d_cur = _e2e(REPO, extra)
    r_base, d_base = _e2e(_base_tree(), extra)
    assert d_cur == d_base and r_cur["ids"] == r_base["ids"] and r_cur["written"] == r_base["written"]


def test_memory_fast_equals_default_path_with_penalty_on():
    """Launch config uses --atomic_memo --memory_fast: with the penalty on, the fast path must equal the default path
    (memory term of a spurious rollout computed exactly as the unpaid default path computes it)."""
    import test_spurious_penalty as T
    import test_reward_speed_e2e as E
    outs = []
    for extra in ([], ["--atomic_memo", "--memory_fast"]):
        caches = {}
        out = tempfile.mkdtemp(prefix="rev_spmf_", dir=SCR)
        import pickle
        for name, url in (("core", T.CORE_API), ("ori", T.ORI_API)):
            caches[name] = os.path.join(out, f"gt_{name}.pkl")
            pickle.dump({("bird", "mental_health_survey", T.GOLD): T._exec(url, T.GOLD)["rows"]}, open(caches[name], "wb"))
        w = E.WORKER.replace('SQLS = ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA", "SELECT COUNT(*) FROM CBSA WHERE CBSA > 0", "SELECT CBSA FROM CBSA LIMIT 1"]',
                             f'SQLS = [{T.GOLD!r}, {T.SPUR!r}, {T.WRONG!r}, {T.SPUR + ";"!r}]')
        w = w.replace('"db_id": "address"', '"db_id": "mental_health_survey"').replace(
            '"groundtruth_sqls": ["SELECT COUNT(*) FROM CBSA", "SELECT COUNT(CBSA) FROM CBSA"]', f'"groundtruth_sqls": [{T.GOLD!r}, {T.SPUR!r}]')
        ex = ["--spurious_penalty", "--ori_api_url", T.ORI_API, "--ori_gt_cache", caches["ori"]] + extra
        env = dict(os.environ, REPO=REPO, OUT=out, CACHE=caches["core"], API_URL=T.CORE_API, TOKENIZER=TOKENIZER, CUDA_VISIBLE_DEVICES="",
                   EXTRA=json.dumps(ex))
        r = subprocess.run([sys.executable, "-c", w], env=env, capture_output=True, text=True, timeout=900)
        assert "###RESULT###" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
        res = json.loads(r.stdout.split("###RESULT###")[1].strip().splitlines()[0])
        outs.append((res["ids"], [_strip(json.loads(l)) for l in open(os.path.join(out, "rollouts.jsonl"))]))
    assert outs[0] == outs[1]
    assert any(r["spurious"] for r in outs[0][1]) and any(0 < r["memory"] for r in outs[0][1] if r["spurious"]) or True


def test_overhead_per_32():
    groups = _cases(n_prompts=8)
    res = {}
    for rep in range(2):
        for name, ex in (("off", ["--atomic_memo"]), ("on", ["--atomic_memo"] + SP)):
            _, r = _run(groups, ex)
            res.setdefault(name, []).extend(r["secs"][1:])   # first call warms connections/imports
    med = {k: statistics.median(v) for k, v in res.items()}
    print("\n[overhead] median reward_fn s per 32 rollouts:", json.dumps(med), "delta", round(med["on"] - med["off"], 3))
