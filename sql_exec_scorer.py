from __future__ import annotations
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore


class SQLExecScorer:
    """
    Score similarity between two SQL execution results (GT vs Prediction) in [0, 1].

    Design:
      1) STRICT COLUMN GATE
         If the set of column names in prediction != GT, score = 0.0.
         (This enforces selecting the correct columns; same count but wrong set => 0.)

      2) ROW CANONICALIZATION
         Convert each row into a normalized tuple (numeric tolerance, lowercased strings)
         so we can compare equality robustly across formatting differences.

      3) COVERAGE (unordered correctness)
         Multiset overlap of rows divided by max(|GT|, |Pred|).
         Penalizes missing/extra rows regardless of order. In [0,1].

      4) ORDER METRIC (on common rows only)
         Compare ranks of the duplicate-aware common rows:
           - mode="rankcorr": Spearman ρ normalized to [0,1]
           - mode="footrule": 1 − normalized L1 distance on ranks
         Gives partial credit when rows are correct but out of order.

      5) FINAL SCORE
         R = cov^gamma * (alpha * order + (1 - alpha)), clipped to [0,1].
         - alpha ∈ [0,1]: weight on order vs unordered correctness
         - gamma ≥ 1: harsher penalty for poor coverage
    """

    def __init__(self, *, alpha: float = 0.5, gamma: float = 1.2,
                 mode: str = "rankcorr", numeric_eps: float = 1e-6):
        """
        Args:
            alpha: [0..1] weight for order term; 0=ignore order, 1=only order (still gated by coverage^gamma).
            gamma: ≥1.0; larger values penalize missing/extra rows more aggressively via coverage^gamma.
            mode : "rankcorr" (Spearman ρ) or "footrule" (L1 rank distance). Both return [0,1].
            numeric_eps: numeric equality tolerance; values are rounded to ~-log10(eps) decimals.

        Raises:
            AssertionError if mode is not supported.
        """
        assert mode in ("rankcorr", "footrule")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.mode = mode
        self.numeric_eps = float(numeric_eps)

    # ---------- Public API ----------
    def score_from_rows(self,
                        gt_rows: Optional[List[Dict[str, Any]]],
                        pred_rows: Optional[List[Dict[str, Any]]]) -> float:
        """
        Compute the score given raw row objects (list of dicts) from the executor.

        The method:
          - Enforces strict column gate (returns 0 if failed).
          - Normalizes data and builds row sequences for GT and Prediction.
          - Computes coverage and order metrics.
          - Combines them into a final score in [0,1].

        Args:
            gt_rows: Ground-truth rows, list of dicts (each dict: col -> value).
            pred_rows: Predicted rows, same structure as gt_rows.

        Returns:
            float in [0,1]. Returns 0 if inputs are None/invalid, columns differ,
            or if either side has no rows (coverage handles empty vs non-empty).
        """
        gt_df = _rows_to_dataframe(gt_rows)
        pr_df = _rows_to_dataframe(pred_rows)
        if gt_df is None or pr_df is None:
            return 0.0

        # (1) Strict column gate: require the same set of columns.
        gt_cols = list(gt_df.columns)
        pr_cols = list(pr_df.columns)
        if set(gt_cols) != set(pr_cols):
            return 0.0

        # Align prediction to GT column order to make row tuples comparable.
        pr_df = pr_df[gt_cols]

        # (2) Canonicalize each row into a normalized tuple.
        gt_seq = self._df_to_row_seq(gt_df, gt_cols)
        pr_seq = self._df_to_row_seq(pr_df, gt_cols)

        # (3) Unordered correctness: multiset overlap / max(len(GT), len(Pred)).
        cov = self._coverage(gt_seq, pr_seq)  # ∈ [0,1]

        # (4) Order on common rows: rank correlation / normalized footrule.
        ord_score = self._order_score(gt_seq, pr_seq, mode=self.mode)  # ∈ [0,1]

        # (5) Final score with coverage harshness (gamma) and order weight (alpha).
        R = (cov ** self.gamma) * (self.alpha * ord_score + (1.0 - self.alpha))
        return max(0.0, min(1.0, float(R)))

    # ---------- Internals ----------
    def _normalize_value(self, v: Any):
        """
        Normalize cell values for robust equality:
          - Numerics: round to ~decimals implied by numeric_eps.
          - Strings/others: lowercased, stripped string form.
        """
        if isinstance(v, (int, float)):
            return round(float(v), self._decimals_for_eps(self.numeric_eps))
        s = str(v)
        # Attempt numeric parse for numeric-like strings ("3.0", "1e-6", etc.)
        try:
            f = float(s)
            return round(f, self._decimals_for_eps(self.numeric_eps))
        except Exception:
            pass
        return s.strip().lower()

    @staticmethod
    def _decimals_for_eps(eps: float) -> int:
        """
        Convert eps (e.g., 1e-6) into a practical rounding precision.
        Caps at 9 decimals to avoid floating noise explosions.
        """
        if eps <= 0:
            return 6
        import math
        d = max(0, int(round(-math.log10(eps))))
        return min(9, max(0, d))

    def _df_to_row_seq(self, df: pd.DataFrame, cols: List[str]) -> List[tuple]:
        """
        Convert a DataFrame slice (cols already aligned) into a list of normalized row tuples.
        """
        rows = []
        for _, r in df.iterrows():
            tup = tuple(self._normalize_value(r[c]) for c in cols)
            rows.append(tup)
        return rows

    @staticmethod
    def _coverage(a_seq: List[tuple], b_seq: List[tuple]) -> float:
        """
        Multiset overlap of rows divided by max(len(a), len(b)).
        - Rewards exact row matches (after normalization).
        - Penalizes both missing and spurious rows.
        """
        from collections import Counter
        A, B = Counter(a_seq), Counter(b_seq)
        inter = sum(min(A[k], B[k]) for k in A.keys() & B.keys())
        denom = max(len(a_seq), len(b_seq), 1)
        return inter / float(denom)

    @staticmethod
    def _make_occ_keys(seq: List[tuple]) -> List[tuple]:
        """
        Expand duplicate rows into unique tokens using occurrence indices.
        Example: ('x',) repeated 3 times → ('x',)#0, ('x',)#1, ('x',)#2.
        This guarantees a consistent 1–1 mapping for rank-based metrics.
        """
        from collections import defaultdict
        counts = defaultdict(int)
        out = []
        for t in seq:
            k = (t, counts[t])
            counts[t] += 1
            out.append(k)
        return out

    def _common_unique_tokens(self, a_seq: List[tuple], b_seq: List[tuple]):
        """
        Build duplicate-aware tokens for both sequences and take their intersection.
        Returns:
            A: tokenized prediction sequence
            B: tokenized GT sequence
            common: set intersection (tokens present in both with the same occurrence index)
        """
        A = self._make_occ_keys(a_seq)
        B = self._make_occ_keys(b_seq)
        common = list(set(A) & set(B))
        return A, B, common

    def _order_score(self, gt_seq: List[tuple], pr_seq: List[tuple], *, mode: str) -> float:
        """
        Compute an order score on the intersection of duplicate-aware tokens.
        If ≤1 common token, return 1.0 (no order information to penalize).
        """
        A, B, common = self._common_unique_tokens(pr_seq, gt_seq)  # A=pred ranks, B=gt ranks
        k = len(common)
        if k <= 1:
            return 1.0

        # 1-based ranks along each sequence
        rankA = {tok: idx for idx, tok in enumerate(A, start=1)}
        rankB = {tok: idx for idx, tok in enumerate(B, start=1)}

        ranksA = [rankA[t] for t in common]
        ranksB = [rankB[t] for t in common]

        if mode == "rankcorr":
            return self._spearman_norm(ranksA, ranksB)   # (ρ+1)/2 in [0,1]
        else:
            return self._footrule_norm(ranksA, ranksB)   # 1 - F/Fmax in [0,1]

    @staticmethod
    def _spearman_norm(rA: List[int], rB: List[int]) -> float:
        """
        Spearman's rho via squared rank differences:
            rho = 1 - (6 * sum (d_i^2)) / (k * (k^2 - 1))
        Then normalized to [0,1] as (rho + 1) / 2.
        """
        k = len(rA)
        if k <= 1:
            return 1.0
        d2 = sum((a - b) * (a - b) for a, b in zip(rA, rB))
        denom = k * (k * k - 1)
        if denom == 0:
            return 1.0
        rho = 1.0 - (6.0 * d2) / float(denom)
        return max(0.0, min(1.0, 0.5 * (rho + 1.0)))

    @staticmethod
    def _footrule_norm(rA: List[int], rB: List[int]) -> float:
        """
        Spearman footrule distance:
            F = sum |rA_i - rB_i|
            Fmax = floor(k^2 / 2)  (max L1 rank distance over permutations)
            ord = 1 - F / Fmax     (normalize to [0,1])
        """
        k = len(rA)
        if k <= 1:
            return 1.0
        F = sum(abs(a - b) for a, b in zip(rA, rB))
        Fmax = (k * k) // 2
        if Fmax <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (F / float(Fmax))))


def _rows_to_dataframe(rows: Optional[List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
    if rows is None:
        return None
    if not isinstance(rows, list):
        return None
    try:
        df = pd.DataFrame(rows)
        return df.fillna("")
    except Exception:
        return None


