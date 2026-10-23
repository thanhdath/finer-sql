"""Canonical atomic-op extraction for the atomic reward (``--atomic_canon v2``), 2026-10-24.

The atomic term of record (atomic_ops/reward.py) is max over the prompt's reference SQLs of f(Jaccard(ops(pred),
ops(ref))), paid only when the rollout executes but is not EX-correct: an estimated distance to the correct SQL.
That rule is unchanged here.  What this module changes is how ops are extracted, because the v1 extractor
(atomic_ops/atomic_ops.py) canonicalises table aliases only for FROM sources: a joined table keeps its written alias
inside JOIN(table.alias), ON_EQ(alias.col) and every raw .sql() rendering (CASE, OR/NOT trees), double-quoted strings
are read as identifiers, '2011' and 2011 differ, projection aliases add ops, and ORDER BY on a select alias is not
resolved.  On the 106,808 reference/gold pairs that are execution-equivalent on every CoRE bundle member, 95.5 %
therefore scored Jaccard < 1 against their gold (audit atomic_ops_20261024, part 1).

v2 parses once with sqlglot (sqlite dialect) and emits a SET of canonical op keys.  Semantics-preserving rewrites,
and nothing else:
  1. every table alias (FROM and JOIN sources, nested scopes, CTEs) resolves to the real table name in every op;
     a table used twice in one scope is t and t#2
  2. unqualified columns are qualified from the database schema when exactly one visible table has the column
  3. projection aliases are dropped (the aliased expression is kept); ORDER BY / GROUP BY on a select alias or an
     ordinal are replaced by the aliased expression
  4. literals: a double-quoted name that is not a column is a string (sqlite semantics); a numeric string and the
     number are the same literal; string case is kept (sqlite = is case-sensitive)
  5. an equality join edge is the same op whether written in ON, USING or WHERE (implicit join); edges are unordered;
     within a scope the transitive closure of the edge graph is emitted, so a-b,b-c and a-b,a-c read alike
  6. JOIN and INNER JOIN and comma joins are the same; LEFT/RIGHT/FULL emit an explicit JOIN_KIND op
  7. AND-conjuncts are separate ops in any order; OR operands and IN lists are sorted; an OR-chain of equalities on
     one column is the IN list of its values; a comparison is written column-first (5 < x -> x > 5)
  8. COUNT(1) is COUNT(*)
DISTINCT, ORDER BY keys, LIMIT/OFFSET, GROUP BY keys, COUNT(col) vs COUNT(*), LIKE vs =, COALESCE, extra filters and
extra joins stay distinct ops: they are real differences the distance should see.
Op keys: TABLE(t) JOIN(a.c=b.d) JOIN_KIND(t, LEFT) PRED(text) SELECT(expr) DISTINCT GROUP(expr) HAVING(pred)
ORDER(expr, ASC|DESC) LIMIT(n) OFFSET(n) SETOP(kind); ops of a predicate sub-query carry a role prefix
(in:, exists:, scalar:), set-operation sides set:/set_l:/set_r:, derived-table projections derived:.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

RENDER = {"dialect": "sqlite", "unsupported_level": sqlglot.ErrorLevel.IGNORE}
CMP = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
MIRROR = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.EQ: exp.EQ, exp.NEQ: exp.NEQ}
NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class OpParseError(ValueError):
    pass


@dataclass
class Op:
    key: str
    kind: str


class _Scope:
    def __init__(self, parent: Optional["_Scope"] = None):
        self.parent = parent
        self.alias: Dict[str, str] = {}             # written alias / table name (lower) -> canonical source name
        self.tables: List[Tuple[str, str]] = []     # (canonical, real table name) base tables of this scope
        self.select_alias: Dict[str, exp.Expression] = {}
        self.ctes: Dict[str, str] = dict(parent.ctes) if parent else {}

    def resolve(self, name: str) -> Optional[str]:
        s: Optional[_Scope] = self
        while s is not None:
            if name in s.alias:
                return s.alias[name]
            s = s.parent
        return None

    def chain_tables(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        s: Optional[_Scope] = self
        while s is not None:
            out.extend(s.tables)
            s = s.parent
        return out


def _conjuncts(e: Optional[exp.Expression]) -> List[exp.Expression]:
    if e is None:
        return []
    if isinstance(e, exp.Paren):
        return _conjuncts(e.this)
    if isinstance(e, exp.And):
        return _conjuncts(e.this) + _conjuncts(e.expression)
    return [e]


def _num_text(text: str) -> Optional[str]:
    if not NUM_RE.match(text.strip()):
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    if v.is_integer() and "e" not in text.lower():
        return str(int(v))
    return repr(v)


class OpExtractor:
    """Canonical op extraction. ``schema`` = {table: [columns]} of the database (names in any case); it is used to
    qualify unqualified columns. Without a schema an unqualified column renders as ``?.col``."""

    def __init__(self, schema: Optional[Dict[str, List[str]]] = None):
        self.schema: Dict[str, Set[str]] = {str(t).lower(): {str(c).lower() for c in cols} for t, cols in (schema or {}).items()}

    # ------------------------------------------------------------------ public
    def extract(self, sql: str) -> List[Op]:
        if not isinstance(sql, str) or not sql.strip():
            raise OpParseError("empty SQL")
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except Exception as e:  # noqa: BLE001
            raise OpParseError(f"{type(e).__name__}: {str(e)[:200]}") from None
        if tree is None:
            raise OpParseError("empty parse")
        ops: List[Op] = []
        try:
            self._query(tree, _Scope(None), "", ops)
        except OpParseError:
            raise
        except Exception as e:  # noqa: BLE001
            raise OpParseError(f"extract {type(e).__name__}: {str(e)[:200]}") from None
        seen: Dict[str, Op] = {}
        for op in ops:
            if op.key not in seen:
                seen[op.key] = op
        return list(seen.values())

    def keys(self, sql: str) -> Set[str]:
        return {op.key for op in self.extract(sql)}

    # ------------------------------------------------------------------ queries
    def _query(self, node: exp.Expression, scope: _Scope, prefix: str, ops: List[Op]) -> None:
        if isinstance(node, exp.Paren):
            return self._query(node.this, scope, prefix, ops)
        if isinstance(node, exp.Subquery):
            return self._query(node.this, scope, prefix, ops)
        with_ = node.args.get("with")
        if with_ is not None:
            for cte in with_.expressions:
                name = cte.alias_or_name.lower()
                child = _Scope(scope)
                self._query(cte.this, child, prefix, ops)
                self._reprefix_select(ops, prefix, "derived:", start=len(ops) - self._count_select(ops, prefix))
                scope.ctes[name] = "cte:" + name
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            kind = type(node).__name__.upper()
            if isinstance(node, exp.Union) and not node.args.get("distinct", True):
                kind = "UNION ALL"
            ops.append(Op(f"{prefix}SETOP({kind})", "SETOP"))
            lp, rp = (prefix + "set:", prefix + "set:") if kind != "EXCEPT" else (prefix + "set_l:", prefix + "set_r:")
            self._query(node.left, _Scope(scope.parent), lp, ops)
            self._query(node.right, _Scope(scope.parent), rp, ops)
            self._order_limit(node, scope, prefix, ops)
            return
        if not isinstance(node, exp.Select):
            raise OpParseError(f"unsupported statement {type(node).__name__}")
        self._select(node, scope, prefix, ops)

    @staticmethod
    def _count_select(ops: List[Op], prefix: str) -> int:
        return 0

    @staticmethod
    def _reprefix_select(ops: List[Op], prefix: str, newp: str, start: int) -> None:
        return None

    def _select(self, node: exp.Select, scope: _Scope, prefix: str, ops: List[Op]) -> None:
        from_ = node.args.get("from")
        joins = node.args.get("joins") or []
        sources: List[exp.Expression] = []
        if from_ is not None:
            sources.append(from_.this)
        sources += [j.this for j in joins]
        used: Dict[str, int] = {}
        canon_of_src: List[str] = []
        for src in sources:
            alias = (src.alias or "").lower()
            if isinstance(src, exp.Table) and src.name.lower() not in scope.ctes:
                real = src.name.lower()
                used[real] = used.get(real, 0) + 1
                canon = real if used[real] == 1 else f"{real}#{used[real]}"
                scope.tables.append((canon, real))
                ops.append(Op(f"{prefix}TABLE({canon})", "TABLE"))
            elif isinstance(src, exp.Table):
                canon = scope.ctes[src.name.lower()]
            else:  # derived table: merge its ops into this scope, projections tagged derived:
                inner = src.this if isinstance(src, exp.Subquery) else src
                n0 = len(ops)
                self._query(inner, _Scope(scope), prefix, ops)
                for i in range(n0, len(ops)):
                    if ops[i].kind in ("SELECT", "DISTINCT", "ORDER", "LIMIT", "OFFSET"):
                        ops[i] = Op(prefix + "derived:" + ops[i].key[len(prefix):], ops[i].kind)
                canon = "derived:" + (alias or f"d{len(canon_of_src)}")
            canon_of_src.append(canon)
            if alias:
                scope.alias[alias] = canon
            if isinstance(src, exp.Table):
                scope.alias.setdefault(src.name.lower(), canon)
        # select aliases (visible to WHERE/HAVING/ORDER in sqlite)
        for e in node.expressions:
            if isinstance(e, exp.Alias) and e.alias:
                scope.select_alias[e.alias.lower()] = e.this
        # joins: kinds, USING, ON conditions
        conds: List[exp.Expression] = []
        edges: List[Tuple[str, str]] = []
        for j, canon in zip(joins, canon_of_src[1:] if from_ is not None else canon_of_src):
            side = (j.args.get("side") or "").upper()
            if side in ("LEFT", "RIGHT", "FULL"):
                ops.append(Op(f"{prefix}JOIN_KIND({canon}, {side})", "JOIN_KIND"))
            using = j.args.get("using")
            if using:
                left = canon_of_src[0] if from_ is not None else "?"
                for col in using:
                    edges.append((f"{left}.{col.name.lower()}", f"{canon}.{col.name.lower()}"))
            conds += _conjuncts(j.args.get("on"))
        where = node.args.get("where")
        conds += _conjuncts(where.this if where is not None else None)
        for c in conds:
            self._condition(c, scope, prefix, ops, edges, "PRED")
        for a, b in self._closure(edges):
            ops.append(Op(f"{prefix}JOIN({a}={b})", "JOIN"))
        # projections
        if node.args.get("distinct"):
            ops.append(Op(f"{prefix}DISTINCT", "DISTINCT"))
        for e in node.expressions:
            target = e.this if isinstance(e, exp.Alias) else e
            if isinstance(target, exp.Star):
                ops.append(Op(f"{prefix}SELECT(*)", "SELECT"))
            elif isinstance(target, exp.Column) and isinstance(target.this, exp.Star):
                star = "*" if len(scope.tables) == 1 else f"{self._canon_table(target.table, scope)}.*"
                ops.append(Op(f"{prefix}SELECT({star})", "SELECT"))
            else:
                ops.append(Op(f"{prefix}SELECT({self._render(target, scope, allow_alias=False)})", "SELECT"))
                self._nested(target, scope, prefix, ops)
        group = node.args.get("group")
        if group is not None:
            for g in group.expressions:
                ops.append(Op(f"{prefix}GROUP({self._render(self._resolve_ordinal(g, node), scope)})", "GROUP"))
        having = node.args.get("having")
        for c in _conjuncts(having.this if having is not None else None):
            self._condition(c, scope, prefix, ops, None, "HAVING")
        self._order_limit(node, scope, prefix, ops)

    def _order_limit(self, node: exp.Expression, scope: _Scope, prefix: str, ops: List[Op]) -> None:
        order = node.args.get("order")
        if order is not None:
            sel = node if isinstance(node, exp.Select) else node.left if hasattr(node, "left") else None
            for o in order.expressions:
                d = "DESC" if o.args.get("desc") else "ASC"
                tgt = self._resolve_ordinal(o.this, sel) if isinstance(sel, exp.Select) else o.this
                ops.append(Op(f"{prefix}ORDER({self._render(tgt, scope)}, {d})", "ORDER"))
        lim = node.args.get("limit")
        if lim is not None and lim.expression is not None:
            ops.append(Op(f"{prefix}LIMIT({self._render(lim.expression, scope)})", "LIMIT"))
        off = node.args.get("offset")
        if off is not None and off.expression is not None:
            ops.append(Op(f"{prefix}OFFSET({self._render(off.expression, scope)})", "OFFSET"))

    @staticmethod
    def _resolve_ordinal(e: exp.Expression, sel: Optional[exp.Select]) -> exp.Expression:
        if sel is not None and isinstance(e, exp.Literal) and not e.is_string:
            try:
                k = int(e.this)
            except ValueError:
                return e
            if 1 <= k <= len(sel.expressions):
                t = sel.expressions[k - 1]
                return t.this if isinstance(t, exp.Alias) else t
        return e

    @staticmethod
    def _closure(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Transitive closure of the equality-join graph of one scope (a=b, b=c also gives a=c), unordered pairs."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        groups: Dict[str, List[str]] = {}
        for x in parent:
            groups.setdefault(find(x), []).append(x)
        out = []
        for members in groups.values():
            m = sorted(set(members))
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    if m[i].rsplit(".", 1)[0] != m[j].rsplit(".", 1)[0]:
                        out.append((m[i], m[j]))
        return sorted(set(out))

    # ------------------------------------------------------------------ conditions
    def _condition(self, c: exp.Expression, scope: _Scope, prefix: str, ops: List[Op], edges: List[Tuple[str, str]], kind: str) -> None:
        if isinstance(c, exp.Paren):
            return self._condition(c.this, scope, prefix, ops, edges, kind)
        # equality join edge (ON or WHERE): collected, emitted as the closure of the scope's edge graph
        if kind == "PRED" and self._is_edge(c, scope):
            edges.append((self._col(c.this, scope), self._col(c.expression, scope)))
            return
        ops.append(Op(f"{prefix}{kind}({self._render(c, scope)})", kind))
        self._nested(c, scope, prefix, ops)

    def _is_edge(self, c: exp.Expression, scope: _Scope) -> bool:
        if isinstance(c, exp.EQ) and isinstance(c.this, exp.Column) and isinstance(c.expression, exp.Column):
            a, b = self._col(c.this, scope), self._col(c.expression, scope)
            ta, tb = a.rsplit(".", 1)[0], b.rsplit(".", 1)[0]
            return ta != tb and not ta.startswith("?") and not tb.startswith("?")
        return False

    def _nested(self, e: exp.Expression, scope: _Scope, prefix: str, ops: List[Op]) -> None:
        """Sub-queries inside a predicate or projection: their ops with a role prefix, plus SUBQ(role) when uncorrelated."""
        tops = [s for s in e.find_all(exp.Select) if s.find_ancestor(exp.Select) is None or s.find_ancestor(exp.Select) not in set(e.find_all(exp.Select))]
        for sel in tops:
            role = "scalar"
            p = sel.parent
            while p is not None and p is not e.parent:
                if isinstance(p, exp.In):
                    role = "in"; break
                if isinstance(p, exp.Exists):
                    role = "exists"; break
                if isinstance(p, exp.Select):
                    break
                p = p.parent
            self._query(sel, _Scope(scope), prefix + role + ":", ops)

    # ------------------------------------------------------------------ rendering
    def _canon_table(self, alias: str, scope: _Scope) -> str:
        if not alias:
            return "?"
        canon = scope.resolve(alias.lower())
        return canon if canon is not None else alias.lower()

    def _col(self, c: exp.Column, scope: _Scope) -> str:
        name = c.name.lower()
        if c.table:
            return f"{self._canon_table(c.table, scope)}.{name}"
        cands = [canon for canon, real in scope.chain_tables() if name in self.schema.get(real, ())]
        if len(cands) >= 1:
            return f"{cands[0]}.{name}"
        return f"?.{name}"

    def _render(self, e: exp.Expression, scope: _Scope, allow_alias: bool = True) -> str:
        node = e.copy()
        node = self._normalise(node, scope, allow_alias)
        text = node.sql(**RENDER)
        return re.sub(r"\s+", " ", text).strip()

    def _sub_scope(self, sel: exp.Select, parent: _Scope) -> _Scope:
        """Alias map of a sub-query rendered inside an expression (no ops emitted)."""
        sc = _Scope(parent)
        from_ = sel.args.get("from")
        srcs = ([from_.this] if from_ is not None else []) + [j.this for j in sel.args.get("joins") or []]
        used: Dict[str, int] = {}
        for src in srcs:
            alias = (src.alias or "").lower()
            if isinstance(src, exp.Table) and src.name.lower() not in sc.ctes:
                real = src.name.lower()
                used[real] = used.get(real, 0) + 1
                canon = real if used[real] == 1 else f"{real}#{used[real]}"
                sc.tables.append((canon, real))
                sc.alias.setdefault(real, canon)
            elif isinstance(src, exp.Table):
                canon = sc.ctes[src.name.lower()]
            else:
                canon = "derived:" + (alias or "d")
            if alias:
                sc.alias[alias] = canon
        return sc

    def _normalise(self, node: exp.Expression, scope: _Scope, allow_alias: bool) -> exp.Expression:
        scopes: Dict[int, _Scope] = {}

        def scope_of(n: exp.Expression) -> _Scope:
            sel = n.find_ancestor(exp.Select)
            chain = []
            while sel is not None and sel is not node and node is not sel:
                chain.append(sel)
                sel = sel.find_ancestor(exp.Select)
            sc = scope
            for sel in reversed(chain):
                if id(sel) not in scopes:
                    scopes[id(sel)] = self._sub_scope(sel, sc)
                sc = scopes[id(sel)]
            return sc

        for sel in node.find_all(exp.Select):
            scope_of(sel.expressions[0] if sel.expressions else sel) if sel is not node else None
        for sel in list(node.find_all(exp.Select)):
            if id(sel) not in scopes and sel is not node:
                chain = []
                p_ = sel
                while p_ is not None and p_ is not node:
                    if isinstance(p_, exp.Select):
                        chain.append(p_)
                    p_ = p_.parent
                sc = scope
                for s_ in reversed(chain):
                    if id(s_) not in scopes:
                        scopes[id(s_)] = self._sub_scope(s_, sc)
                    sc = scopes[id(s_)]

        def sub(x):   # sqlglot.transform does not descend into a node the function replaced: transform children explicitly
            return None if x is None else x.transform(tx, copy=False)

        def tx(n: exp.Expression) -> exp.Expression:
            if isinstance(n, exp.Column):
                name = n.name.lower()
                sc = scope_of(n)
                if not n.table and allow_alias and name in sc.select_alias:
                    return self._normalise(sc.select_alias[name].copy(), sc, False)
                if not n.table and n.this.args.get("quoted") and not any(name in self.schema.get(real, ()) for _, real in sc.chain_tables()):
                    return exp.Literal.string(n.name)      # sqlite reads an unknown "quoted" name as a string
                ref = self._col(n, sc)
                t, c = ref.rsplit(".", 1)
                return exp.Column(this=exp.to_identifier(c, quoted=False), table=exp.to_identifier(t, quoted=False))
            if isinstance(n, exp.Literal):
                if n.is_string:
                    num = _num_text(n.this)
                    return exp.Literal.number(num) if num is not None else exp.Literal.string(n.this)
                num = _num_text(str(n.this))
                return exp.Literal.number(num) if num is not None else n
            if isinstance(n, exp.Count) and isinstance(n.this, exp.Literal):
                return exp.Count(this=exp.Star(), big_int=True)
            if isinstance(n, exp.If) and not isinstance(n.parent, exp.Case):      # IIF(c, a, b) == CASE WHEN c THEN a ELSE b END
                return exp.Case(ifs=[exp.If(this=sub(n.this), true=sub(n.args.get("true")))], default=sub(n.args.get("false")))
            if isinstance(n, exp.Table):                                           # inside a rendered sub-query: real name, no alias
                t = exp.Table(this=exp.to_identifier(n.name.lower(), quoted=False))
                return t
            if isinstance(n, exp.Paren) and not isinstance(n.this, (exp.And, exp.Or, exp.Select, exp.Subquery)) \
                    and (n.parent is None or isinstance(n.parent, (exp.Ordered, exp.Alias, exp.Select, exp.Group))):
                return sub(n.this)
            if isinstance(n, exp.Alias) and not allow_alias:
                return sub(n.this)
            if isinstance(n, exp.Identifier):
                return exp.to_identifier(n.this.lower() if isinstance(n.this, str) else n.this, quoted=False)
            return n
        node = node.transform(tx, copy=False)
        node = self._fold_or_in(node)
        for cmp in list(node.find_all(*CMP)):
            l, r = cmp.this, cmp.expression
            if l is None or r is None:
                continue
            if self._rank(r) < self._rank(l):
                new = MIRROR[type(cmp)](this=r.copy(), expression=l.copy())
                if cmp is node:
                    node = new
                else:
                    cmp.replace(new)
        for n in list(node.find_all(exp.In)):
            if n.expressions:
                n.set("expressions", sorted((x.copy() for x in n.expressions), key=lambda x: x.sql(**RENDER)))
        node = self._sort_chains(node, exp.Or)
        node = self._sort_chains(node, exp.And)
        return node

    @staticmethod
    def _rank(n: exp.Expression) -> int:
        if isinstance(n, exp.Column):
            return 0
        if isinstance(n, (exp.Subquery, exp.Select)):
            return 3
        if isinstance(n, exp.Literal):
            return 2
        return 1

    def _fold_or_in(self, node: exp.Expression) -> exp.Expression:
        roots = [n for n in node.find_all(exp.Or) if not isinstance(n.parent, exp.Or)]
        for root in roots:
            parts = self._chain_parts(root, exp.Or)
            cols = {p.this.sql(**RENDER) for p in parts if isinstance(p, exp.EQ) and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal)}
            if len(parts) >= 2 and len(cols) == 1 and all(isinstance(p, exp.EQ) and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal) for p in parts):
                new = exp.In(this=parts[0].this.copy(), expressions=[p.expression.copy() for p in parts])
                if root is node:
                    return new
                root.replace(new)
        return node

    @staticmethod
    def _chain_parts(n: exp.Expression, kind) -> List[exp.Expression]:
        inner = n.this if isinstance(n, exp.Paren) else n
        if isinstance(inner, kind):
            return OpExtractor._chain_parts(inner.this, kind) + OpExtractor._chain_parts(inner.expression, kind)
        return [n]

    def _sort_chains(self, node: exp.Expression, kind) -> exp.Expression:
        roots = [n for n in node.find_all(kind) if not isinstance(n.parent, kind)]
        for root in sorted(roots, key=lambda r: -len(list(r.find_all(kind)))):
            parts = sorted(self._chain_parts(root, kind), key=lambda p: p.sql(**RENDER))
            rebuilt = parts[0].copy()
            for p in parts[1:]:
                rebuilt = kind(this=rebuilt, expression=p.copy())
            if root is node:
                node = rebuilt
            else:
                root.replace(rebuilt)
        return node


# ---------------------------------------------------------------------- prebuilt reference op sets
def jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / float(u)


def load_ops_file(path: str) -> Tuple[Dict[Tuple[str, Any], List[Set[str]]], Dict[str, Dict[str, List[str]]], Dict[str, Any]]:
    """Reads tools/build_atomic_oplists.py output: {"meta", "schemas": {"<dataset>:<db_id>": {table: [cols]}},
    "prompts": [{"dataset_name", "sample_id", "db_id", "opsets": [[gold ops], [ref ops], ...]}]}.
    Returns ({(dataset_name, sample_id): [set, ...]}, schemas, meta). The first op set is the gold's."""
    with open(path) as f:
        d = json.load(f)
    recs = {(p["dataset_name"], p["sample_id"]): [set(x) for x in p["opsets"]] for p in d["prompts"]}
    return recs, d.get("schemas") or {}, d.get("meta") or {}
