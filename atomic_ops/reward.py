
from typing import List, Optional, Set
from atomic_ops.atomic_ops import sql_to_ops
import numpy as np

class AtomicOpsReward:
    """Compute Jaccard similarity between atomic ops of two SQL queries."""

    def __init__(self, *, dialect: str = "sqlite", module_path: Optional[str] = None, e: float = 0.30, beta: float = 0.71, gamma: float = 5.0) -> None:
        self.dialect = dialect
        self.e = e
        self.beta = beta
        self.gamma = gamma

    def _ops_set(self, sql: str) -> Set[str]:
        try:
            ops = sql_to_ops(sql, dialect=self.dialect)  # type: ignore[misc]
        except Exception:
            ops = []
        return set(ops)

    # execution-weighted Jaccard with power penalty
    def execution_weighted_jaccard(self, s, e: Optional[float] = None, beta: Optional[float] = None, gamma: Optional[float] = None) -> float:
        s = np.asarray(s)
        e = e if e is not None else self.e
        beta = beta if beta is not None else self.beta
        gamma = gamma if gamma is not None else self.gamma
        return float(e * s + (1.0 - e) * beta * (s ** gamma))

    def score(self, sql_a: str, sql_b: str) -> float:
        """Return Jaccard similarity (|A∩B| / |A∪B|) in [0,1]."""
        a = self._ops_set(sql_a)
        b = self._ops_set(sql_b)
        union = len(a | b)
        if union == 0:
            return 1.0
        inter = len(a & b)
        return inter / float(union)

    def score_against_list(self, pred_sql: str, gt_sqls: List[str]) -> float:
        """Return max Jaccard similarity between pred_sql and any SQL in gt_sqls."""
        pred_ops = self._ops_set(pred_sql)
        best = 0.0
        for s in (gt_sqls or []):
            # norm s, strip() and make sure it ends with semicolon
            s = s.strip()
            if not s.endswith(";"):
                s = s + ";"
            
            gt_ops = self._ops_set(s)
            inter = len(pred_ops & gt_ops)
            union = len(pred_ops | gt_ops)
            # jaccard similarity
            score = 0.0 if union == 0 else inter / float(union)
            score = self.execution_weighted_jaccard(score)
            if score > best:
                best = score

        return float(best)
