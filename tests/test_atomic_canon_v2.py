"""--atomic_canon v2 (atomic_ops/ops_v2.py): canonical op extraction for the atomic term. CPU only, no SQL.

(a) spelling differences that are semantics-preserving (table aliases, projection aliases, column qualifiers, implicit
    vs explicit joins, join-edge transitivity, conjunct order, OR-chain vs IN list, column-first comparisons,
    double-quoted strings, numeric strings) give Jaccard 1.0 against the gold; real differences do not.
(b) the extractor does not throw on any corpus gold (train_hard parquet of the screen, skipped if absent).
(c) the v1 term (AtomicOpsReward.score_against_list) is untouched: fixed pairs give the values recorded when v2 was
    added, and reward.py / atomic_ops.py hash as before.
(d) the ops-file loader returns the gold's op set first.

    python -m pytest tests/test_atomic_canon_v2.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atomic_ops.ops_v2 import OpExtractor, OpParseError, jaccard, load_ops_file  # noqa: E402
from atomic_ops.reward import AtomicOpsReward  # noqa: E402

SCHEMA = {"CreditCard": ["CreditCardID", "CardNumber", "CardType"], "PersonCreditCard": ["BusinessEntityID", "CreditCardID"],
          "customers": ["customer_id", "customer_name"], "customers_policies": ["customer_id", "policy_id"],
          "Draft_Copies": ["document_id", "copy_number"], "exhibition": ["Ticket_Price", "Year"],
          "Master": ["playerID", "shootCatch", "firstNHL"], "Rooms": ["roomName", "bedType"],
          "a": ["name", "x"], "b": ["name"], "c": ["name"]}
EQUIVALENT = [
    # table aliases + column qualifiers + numeric string (audit: predicate_diff / value_spelling)
    ("SELECT T2.BusinessEntityID FROM CreditCard AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.CreditCardID = T2.CreditCardID WHERE T1.CardNumber = 11113366963373",
     "SELECT pc.BusinessEntityID FROM PersonCreditCard AS pc JOIN CreditCard AS cc ON pc.CreditCardID = cc.CreditCardID WHERE cc.CardNumber = '11113366963373';"),
    # implicit join, double-quoted string, projection alias (audit: implicit_explicit_join / projection_alias)
    ('SELECT count(*) FROM customers AS t1 JOIN customers_policies AS t2 ON t1.customer_id  =  t2.customer_id WHERE t1.customer_name  =  "Dayana Robel"',
     "SELECT COUNT(*) AS policy_count FROM Customers c, Customers_Policies cp WHERE cp.Customer_ID = c.Customer_ID AND c.Customer_name = 'Dayana Robel'"),
    # ORDER BY on a select alias (audit: other)
    ("SELECT document_id , count(copy_number) FROM Draft_Copies GROUP BY document_id ORDER BY count(copy_number) DESC LIMIT 1;",
     "SELECT dc.document_id AS document_id, COUNT(dc.copy_number) AS num_draft_copies FROM Draft_Copies dc GROUP BY dc.document_id ORDER BY num_draft_copies DESC LIMIT 1"),
    # projection aliases, keyword case
    ("SELECT avg(ticket_price) ,  min(ticket_price) ,  max(ticket_price) FROM exhibition WHERE YEAR  <  2009",
     "SELECT AVG(Ticket_Price) AS avg_ticket_price, MIN(Ticket_Price) AS min_ticket_price, MAX(Ticket_Price) AS max_ticket_price FROM exhibition WHERE Year < 2009"),
    # conjunct order + numeric string
    ("SELECT COUNT(playerID) FROM Master WHERE shootCatch IS NULL AND firstNHL = '2011'",
     "SELECT COUNT(playerid) FROM master WHERE master.firstnhl = 2011 AND master.shootcatch IS NULL;"),
    # OR-chain of equalities vs IN list, column-first comparison
    ('SELECT roomName FROM Rooms WHERE bedType  =  "King" OR bedType  =  "Queen"',
     "SELECT roomName FROM Rooms WHERE bedType IN ('Queen', 'King')"),
    # join-edge transitivity
    ("SELECT AVG(a.x) FROM a JOIN b ON b.name = a.name JOIN c ON c.name = b.name",
     "SELECT AVG(T1.x) FROM a AS T1 INNER JOIN b AS T2 ON T2.name = T1.name INNER JOIN c AS T3 ON T3.name = T1.name"),
    # ORDER BY ordinal, COUNT(1)
    ("SELECT name, COUNT(1) FROM b GROUP BY name ORDER BY 2 DESC", "SELECT b.name, COUNT(*) FROM b GROUP BY b.name ORDER BY COUNT(*) DESC"),
]
DIFFERENT = [
    ("SELECT COUNT(playerID) FROM Master", "SELECT COUNT(*) FROM Master"),
    ("SELECT roomName FROM Rooms", "SELECT DISTINCT roomName FROM Rooms"),
    ("SELECT roomName FROM Rooms ORDER BY roomName", "SELECT roomName FROM Rooms ORDER BY roomName LIMIT 1"),
    ("SELECT name FROM b", "SELECT b.name FROM b LEFT JOIN c ON c.name = b.name"),
]


def _j(ex, a, b):
    return jaccard(ex.keys(a), ex.keys(b))


def test_equivalent_spellings_score_one():
    ex = OpExtractor(SCHEMA)
    for a, b in EQUIVALENT:
        assert _j(ex, a, b) == 1.0, (a, b, sorted(ex.keys(a) ^ ex.keys(b)))


def test_real_differences_stay_below_one():
    ex = OpExtractor(SCHEMA)
    for a, b in DIFFERENT:
        assert _j(ex, a, b) < 1.0, (a, b)


def test_v1_term_untouched():
    """The default path: same numbers as before v2 existed (recorded 2026-10-24 from atomic_ops.py 464d7e5e / reward.py e836ab39)."""
    r = AtomicOpsReward(dialect="sqlite", e=0.05, beta=0.79, gamma=0.20)
    pred = "SELECT pc.BusinessEntityID FROM PersonCreditCard AS pc JOIN CreditCard AS cc ON pc.CreditCardID = cc.CreditCardID WHERE cc.CardNumber = '11113366963373'"
    gold = "SELECT T2.BusinessEntityID FROM CreditCard AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.CreditCardID = T2.CreditCardID WHERE T1.CardNumber = 11113366963373"
    assert abs(r.score(pred, gold) - 1 / 7) < 1e-12                  # 1 op shared of 7 (v1 alias leakage)
    assert abs(r.score_against_list(pred, [gold]) - (0.05 / 7 + 0.95 * 0.79 * (1 / 7) ** 0.2)) < 1e-12
    assert r.score_against_list(gold, [gold]) == pytest.approx(0.05 + 0.95 * 0.79)
    sha = {f: hashlib.sha256(open(os.path.join(ROOT, "atomic_ops", f), "rb").read()).hexdigest()[:8] for f in ("atomic_ops.py", "reward.py")}
    assert sha == {"atomic_ops.py": "464d7e5e", "reward.py": "e836ab39"}


@pytest.mark.skipif(not os.path.isfile("/home/datht/finer-sql-tkde/experiments/05_core05f_finer_config_20261023/reference_screen_20261023/train_hard_core_screened.parquet"),
                    reason="corpus parquet not present")
def test_extractor_never_throws_on_corpus_golds():
    import pandas as pd
    p = "/home/datht/finer-sql-tkde/experiments/05_core05f_finer_config_20261023/reference_screen_20261023/train_hard_core_screened.parquet"
    df = pd.read_parquet(p, columns=["dataset_name", "db_id", "groundtruth_sqls"])
    sch_path = "/home/datht/finer-sql-tkde/experiments/05_core05f_finer_config_20261023/atomic_ops_20261024/build/schemas_215.json"
    schemas = json.load(open(sch_path)) if os.path.isfile(sch_path) else {}
    exs, fails, n = {}, [], 0
    for ds, db, refs in zip(df.dataset_name, df.db_id, df.groundtruth_sqls):
        ex = exs.setdefault(f"{ds}:{db}", OpExtractor(schemas.get(f"{ds}:{db}")))
        n += 1
        try:
            assert ex.keys(refs[0])
        except (OpParseError, AssertionError) as e:
            fails.append((ds, db, refs[0][:80], str(e)[:80]))
    assert n == 9687 and not fails, fails[:5]


def test_ops_file_loader_gold_first():
    d = {"meta": {}, "schemas": {"bird:x": {"t": ["a"]}},
         "prompts": [{"dataset_name": "bird", "sample_id": 7, "db_id": "x", "opsets": [["TABLE(t)", "SELECT(t.a)"], ["TABLE(t)", "SELECT(t.a)", "DISTINCT"]]}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(d, f)
    recs, schemas, meta = load_ops_file(f.name)
    assert recs[("bird", 7)][0] == {"TABLE(t)", "SELECT(t.a)"} and len(recs[("bird", 7)]) == 2 and schemas["bird:x"] == {"t": ["a"]}
