#!/usr/bin/env python3
"""Every reference must belong to the sample it is written onto.

The defect this guards against: `add_correct_sqls_from_eval.py` used to address a
sample's document by ``{"db_id": db_id, "_id": sample_id}`` -- it assumed the
collection's primary key ``_id`` is the same number as the evaluation's ``sample_id``.
Where those two id spaces disagree (they run parallel inside one database but are offset
by however many of that database's samples one of them dropped), ``_id == sample_id``
selects a DIFFERENT sample of the SAME database, and the correct SQLs of sample N land
on sample N-k.  Measured on the shipped corpus, 73.9% of the mis-attached references sit
at their database's single dominant offset -- bird/airline is +5 throughout, which is why
sample 5849 ("flights that landed at Lake Charles Regional on 2018/8/15") carried eleven
references counting on-time departures on 2018/8/1: sample 5854's.

The tests below run against an in-memory fake collection, so they need no MongoDB.

    python3 -m pytest tests/test_add_correct_sqls_provenance.py -q
    python3 tests/test_add_correct_sqls_provenance.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "evaluation"))
import add_correct_sqls_from_eval as M                                    # noqa: E402


class FakeCollection:
    def __init__(self, docs):
        self.docs = [dict(d) for d in docs]

    def find_one(self, query):
        for d in self.docs:
            if all(d.get(k) == v for k, v in query.items()):
                return d
        return None

    def update_one(self, flt, update, upsert=False):
        d = self.find_one(flt)
        if d is not None:
            d.update(update["$set"])


class FakeClient:
    def __init__(self, coll):
        self._coll = coll

    def __getitem__(self, _db):
        return {"samples": self._coll}


#: bird/airline as the corpus holds it: the two id spaces are offset by +5, so the
#: document whose ``_id`` is 5849 is really sample 5854's.
AIRLINE = [
    {"_id": 5844, "sample_id": 5849, "db_id": "airline", "dataset_name": "bird",
     "SQL": "SELECT COUNT(T1.Code) FROM Airports AS T1 INNER JOIN Airlines AS T2 "
            "ON T1.Code = T2.DEST WHERE T2.FL_DATE = '2018/8/15' "
            "AND T1.Description = 'Lake Charles, LA: Lake Charles Regional'"},
    {"_id": 5849, "sample_id": 5854, "db_id": "airline", "dataset_name": "bird",
     "SQL": "SELECT COUNT(*) FROM Airlines WHERE FL_DATE = '2018/8/1' "
            "AND DEP_DELAY <= 0"},
]

ENTRY_5854 = {
    "sample_id": 5854, "db_id": "airline", "dataset_name": "bird",
    "ground_truth_sql": AIRLINE[1]["SQL"],
    "gt_execution_result": {"ok": True, "rows": [{"c": 3}]},
    "predicted_sqls": [
        "SELECT SUM(CASE WHEN dep_delay <= 0 THEN 1 ELSE 0 END) "
        "FROM airlines WHERE fl_date = '2018/8/1';"],
    "candidate_execution_results": [{"ok": True, "rows": [{"c": 3}]}],
}


def _plan(docs, entries):
    coll = FakeCollection(docs)
    planned, stats = M.build_planned_updates(entries, FakeClient(coll), "mats", "samples")
    return coll, planned, stats


def test_references_go_to_their_own_sample_not_the_id_neighbour():
    coll, planned, _stats = _plan(AIRLINE, [ENTRY_5854])
    assert len(planned) == 1, "the sample's own document must be found"
    assert planned[0]["document_id"] == 5849, \
        "must address the document whose sample_id is 5854, not the one whose _id is 5854"
    M.execute_updates(planned, FakeClient(coll), "mats", "samples")
    by_sample = {d["sample_id"]: d for d in coll.docs}
    assert M.FIELD_NAME not in by_sample[5849], \
        "sample 5849 (Lake Charles) must NOT receive sample 5854's dep_delay references"
    refs = by_sample[5854][M.FIELD_NAME]
    assert refs[0] == M.canonicalize_for_storage(ENTRY_5854["ground_truth_sql"])
    assert any("dep_delay" in r.lower() for r in refs[1:])


def test_every_reference_belongs_to_its_sample():
    """The invariant, stated directly: same database, and the document's own gold is the
    gold the reference was judged equal to."""
    coll, planned, _stats = _plan(AIRLINE, [ENTRY_5854])
    M.execute_updates(planned, FakeClient(coll), "mats", "samples")
    for d in coll.docs:
        refs = d.get(M.FIELD_NAME)
        if not refs:
            continue
        assert M.normalize_for_compare(refs[0]) == M.normalize_for_compare(d["SQL"]), \
            "entry 0 must be this document's own gold"
        for entry in [ENTRY_5854]:
            if entry["sample_id"] == d["sample_id"]:
                assert entry["db_id"] == d["db_id"]
                assert M.normalize_for_compare(entry["ground_truth_sql"]) == \
                    M.normalize_for_compare(d["SQL"])


def test_a_document_whose_gold_disagrees_is_refused():
    """Even if some third id space is introduced later, a wrong document is not written."""
    docs = [{"_id": 1, "sample_id": 5854, "db_id": "airline", "dataset_name": "bird",
             "SQL": "SELECT COUNT(*) FROM Airports"}]
    coll, planned, stats = _plan(docs, [ENTRY_5854])
    assert planned == [], "a gold mismatch must refuse the write"
    assert stats["entries_gold_mismatch"] == 1
    M.execute_updates(planned, FakeClient(coll), "mats", "samples")
    assert M.FIELD_NAME not in coll.docs[0]


def test_missing_document_is_counted_not_guessed():
    coll, planned, stats = _plan([], [ENTRY_5854])
    assert planned == [] and stats["entries_no_document"] == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except AssertionError as exc:
                fails += 1
                print("FAIL %s: %s" % (name, exc))
    raise SystemExit(1 if fails else 0)
