"""
sql2ops_alias.py
----------------
Convert a SQL string into a deterministic list of atomic ops.
- Uses sqlglot to parse.
- Canonicalizes table aliases per query context: T1, T2, T3... (per subquery).
- Every Column/Table reference is emitted with canonical alias when available.

pip install sqlglot
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import sqlglot
from sqlglot import exp

AGG_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}

# ------------------------------ Public API ------------------------------ #
def sql_to_ops(sql: str, dialect: str = "duckdb") -> List[str]:
    root = sqlglot.parse_one(sql, read=dialect)
    ops: List[str] = []
    _emit_query_ops(root, ops, alias_map_stack=[{}])  # new alias scope
    return ops

# ------------------------------ Emitters -------------------------------- #
def _emit_query_ops(node: exp.Expression, ops: List[str], alias_map_stack: List[Dict[str, str]]):
    # WITH / CTEs
    if isinstance(node, exp.With):
        for c in node.find_all(exp.CTE):
            name = c.alias_or_name
            ops.append(f"WITH_CTE({name})")
            ops.append("ENTER_SUBQUERY(CTE)")
            alias_map_stack.append({})
            _emit_query_ops(c.this, ops, alias_map_stack)
            alias_map_stack.pop()
            ops.append("EXIT_SUBQUERY")
        _emit_query_ops(node.this, ops, alias_map_stack)
        return

    # FROM block — declare canonical aliases for base + joins (this scope)
    _emit_from_and_canon_aliases(node, ops, alias_map_stack)

    # JOINs (and ON)
    for j in node.find_all(exp.Join, bfs=False):
        jt = (j.args.get("kind") or "INNER").upper()
        t_name, t_alias_key = _fmt_table_and_key(j.this)
        # Emit JOIN without alias rendering (no "as <alias>")
        ops.append(f"JOIN({t_name}, {jt})")
        on_expr = j.args.get("on")
        if on_expr:
            _emit_on(on_expr, ops, alias_map_stack)

    # SELECT
    _emit_select(node, ops, alias_map_stack)

    # WHERE
    if node.args.get("where"):
        _emit_pred_block(node.args["where"].this, ops, "WHERE_PRED", alias_map_stack, show_table=False)

    # GROUP BY → HAVING
    if gb := node.args.get("group"):
        for e in gb.expressions:
            ops.append(f"GROUP_BY({_fmt_expr(e, alias_map_stack, show_table=False)})")
    if hv := node.args.get("having"):
        _emit_pred_block(hv.this, ops, "HAVING_PRED", alias_map_stack, show_table=False)

    # ORDER → LIMIT
    if ob := node.args.get("order"):
        for o in ob.expressions:
            dir_ = "DESC" if o.args.get("desc") else "ASC"
            ops.append(f"ORDER_BY({_fmt_expr(o.this, alias_map_stack, show_table=False)}, {dir_})")
    if lm := node.args.get("limit"):
        n = lm.expression
        if isinstance(n, exp.Literal) and n.is_int:
            ops.append(f"LIMIT({n.this})")
        else:
            ops.append(f"LIMIT({_fmt_expr(n, alias_map_stack, show_table=False)})")

    # Set ops (if this node itself is a set-op)
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        _emit_query_ops(node.left, ops, alias_map_stack)
        op = node.key.upper() + ("" if node.args.get("distinct", True) else "_ALL")
        ops.append(op)
        ops.append("ENTER_SUBQUERY(SET_RIGHT)")
        alias_map_stack.append({})
        _emit_query_ops(node.right, ops, alias_map_stack)
        alias_map_stack.pop()
        ops.append("EXIT_SUBQUERY")


def _emit_from_and_canon_aliases(node: exp.Expression, ops: List[str], alias_map_stack: List[Dict[str, str]]):
    from_ = node.args.get("from")
    if not from_:
        return

    # First pass: assign canonical aliases in order of appearance (T1, T2, …)
    scope = alias_map_stack[-1]
    next_id = 1

    def register_alias(key: str) -> str:
        nonlocal next_id
        if key not in scope:
            scope[key] = f"T{next_id}"
            next_id += 1
        return scope[key]

    # Emit FROM sources (tables or subqueries)
    for src in from_.expressions:
        if isinstance(src, exp.Subquery):
            ops.append("ENTER_SUBQUERY(FROM)")
            alias_map_stack.append({})
            _emit_query_ops(src.this, ops, alias_map_stack)
            alias_map_stack.pop()
            ops.append("EXIT_SUBQUERY")
            alias_key = (src.alias and src.alias) or "SUBQUERY"
            canon = register_alias(str(alias_key))
            ops.append(f"FROM(SUBQUERY as {canon})")
        elif isinstance(src, exp.Select):
            ops.append("ENTER_SUBQUERY(FROM)")
            alias_map_stack.append({})
            _emit_query_ops(src, ops, alias_map_stack)
            alias_map_stack.pop()
            ops.append("EXIT_SUBQUERY")
            canon = register_alias("SUBQUERY")
            ops.append(f"FROM(SUBQUERY as {canon})")
        else:
            t_name, t_alias_key = _fmt_table_and_key(src)
            canon = register_alias(t_alias_key)
            ops.append(f"FROM({t_name} as {canon})")


def _emit_on(cond: exp.Expression, ops: List[str], alias_map_stack: List[Dict[str, str]]):
    if isinstance(cond, exp.And):
        for c in _flatten_ands(cond):
            _emit_on(c, ops, alias_map_stack)
        return
    if isinstance(cond, exp.EQ):
        ops.append(f"ON_EQ({_fmt_expr(cond.left, alias_map_stack)}, {_fmt_expr(cond.right, alias_map_stack)})")
        return
    if isinstance(cond, (exp.Or, exp.Not)):
        ops.append(f"ON_PRED_TREE({cond.sql()})")
        return
    # Fallback for any predicate-like node; guard against non-Expression inputs
    if not isinstance(cond, exp.Expression):
        ops.append(f"ON_PRED_TREE({repr(cond)})")
        return
    ops.append(_fmt_pred("ON_PRED", cond, alias_map_stack))


def _emit_select(node: exp.Expression, ops: List[str], alias_map_stack: List[Dict[str, str]]):
    sel = node if isinstance(node, exp.Select) else node.find(exp.Select, bfs=False)
    if not isinstance(sel, exp.Select):
        return

    for proj in sel.expressions:
        alias_name = None
        target = proj
        if isinstance(proj, exp.Alias):
            alias_name = proj.alias
            target = proj.this

        # Windowed function
        if isinstance(target, exp.Func) and target.args.get("over"):
            func_name = target.name.upper()
            _args_list = []
            exprs = target.args.get("expressions")
            if exprs:
                _args_list.extend(exprs)
            if target.args.get("this") is not None:
                _args_list.append(target.args.get("this"))
            arg_str = ", ".join(_fmt_expr(a, alias_map_stack, show_table=False) for a in _args_list)
            win = target.args["over"]
            ops.append(f"WINDOW({_fmt_window_spec(win, alias_map_stack)})")
            ops.append(f"SELECT_WIN({func_name}, {arg_str})")
            if alias_name:
                ops.append(f"ALIAS(COLUMN, {func_name}({arg_str}), {alias_name})")
            continue

        # Aggregates vs plain columns vs expressions
        if isinstance(target, exp.Func) and target.name and target.name.upper() in AGG_FUNCS:
            func = target.name.upper()
            _args_list = []
            exprs = target.args.get("expressions")
            if exprs:
                _args_list.extend(exprs)
            if target.args.get("this") is not None:
                _args_list.append(target.args.get("this"))
            args = ", ".join(_fmt_expr(a, alias_map_stack, show_table=False) for a in _args_list)
            ops.append(f"SELECT_AGG({func}, {args})")
            if alias_name:
                ops.append(f"ALIAS(COLUMN, {func}({args}), {alias_name})")
        elif isinstance(target, exp.Column):
            ops.append(f"SELECT_COL({_fmt_expr(target, alias_map_stack, show_table=False)})")
            if alias_name:
                ops.append(f"ALIAS(COLUMN, {_fmt_expr(target, alias_map_stack, show_table=False)}, {alias_name})")
        elif isinstance(target, exp.Star):
            ops.append("SELECT_COL(*)")
            if alias_name:
                ops.append(f"ALIAS(COLUMN, *, {alias_name})")
        else:
            s = _fmt_func_or_expr(target, alias_map_stack, show_table=False)
            ops.append(f"SELECT_EXPR({s})")
            if alias_name:
                ops.append(f"ALIAS(COLUMN, {s}, {alias_name})")

    if sel.args.get("distinct"):
        ops.append("DISTINCT")


def _emit_pred_block(expr_: exp.Expression, ops: List[str], kind: str, alias_map_stack: List[Dict[str, str]], show_table: bool = True):
    for p in _flatten_ands(expr_):
        if isinstance(p, (exp.Or, exp.Not)):
            ops.append(f"{kind}_TREE({p.sql()})")
            continue

        # ---- NEW: handle scalar subqueries in binary predicates ----
        # Derive role name from 'kind'
        role = "WHERE" if kind == "WHERE_PRED" else ("HAVING" if kind == "HAVING_PRED" else "PRED")

        if isinstance(p, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            op = p.key.upper()
            L, R = p.left, p.right

            # Right side is a scalar subquery:  col OP (SELECT ...)
            if isinstance(R, exp.Subquery):
                lhs = _fmt_expr(L, alias_map_stack, show_table=show_table)
                ops.append(f"ENTER_SUBQUERY({role}_SCALAR)")
                alias_map_stack.append({})
                _emit_query_ops(R.this, ops, alias_map_stack)
                alias_map_stack.pop()
                ops.append("EXIT_SUBQUERY")
                ops.append(f"{kind}({op}, {lhs}, SUBQ_LAST)")
                continue

            # Left side is a scalar subquery:  (SELECT ...) OP expr
            if isinstance(L, exp.Subquery):
                rhs = _fmt_expr(R, alias_map_stack, show_table=show_table)
                ops.append(f"ENTER_SUBQUERY({role}_SCALAR)")
                alias_map_stack.append({})
                _emit_query_ops(L.this, ops, alias_map_stack)
                alias_map_stack.pop()
                ops.append("EXIT_SUBQUERY")
                ops.append(f"{kind}({op}, SUBQ_LAST, {rhs})")
                continue
        # ---- END NEW ----

        if isinstance(p, exp.In) and isinstance(p.args.get("query"), exp.Subquery):
            col = _fmt_expr(p.this, alias_map_stack, show_table=show_table)
            ops.append("ENTER_SUBQUERY(WHERE_IN)")
            alias_map_stack.append({})
            _emit_query_ops(p.args["query"].this, ops, alias_map_stack)
            alias_map_stack.pop()
            ops.append("EXIT_SUBQUERY")
            ops.append(f"{kind}(IN_SUBQ, {col}, SUBQ_LAST)")
            continue

        if isinstance(p, exp.Exists):
            ops.append("ENTER_SUBQUERY(WHERE_EXISTS)")
            alias_map_stack.append({})
            _emit_query_ops(p.this, ops, alias_map_stack)
            alias_map_stack.pop()
            ops.append("EXIT_SUBQUERY")
            ops.append(f"{kind}(EXISTS_SUBQ, TRUE, SUBQ_LAST)")
            continue

        ops.append(_fmt_pred(kind, p, alias_map_stack, show_table=show_table))


# ------------------------------ Helpers --------------------------------- #
def _flatten_ands(e: exp.Expression) -> List[exp.Expression]:
    if isinstance(e, exp.And):
        out: List[exp.Expression] = []
        out.extend(_flatten_ands(e.left))
        out.extend(_flatten_ands(e.right))
        return sorted(out, key=lambda x: x.sql())
    return [e]

def _fmt_pred(kind: str, p: exp.Expression, alias_map_stack: List[Dict[str, str]], show_table: bool = True) -> str:
    # Defensive guard: p might be a raw python object if upstream produced non-Expression
    if not isinstance(p, exp.Expression):
        return f"{kind}_TREE({repr(p)})"
    if isinstance(p, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike)):
        op = p.key.upper()
        return f"{kind}({op}, {_fmt_expr(p.left, alias_map_stack, show_table=show_table)}, {_fmt_expr(p.right, alias_map_stack, show_table=show_table)})"
    if isinstance(p, exp.Between):
        return f"{kind}(BETWEEN, {_fmt_expr(p.this, alias_map_stack, show_table=show_table)}, [{_fmt_expr(p.args['low'], alias_map_stack, show_table=show_table)}, {_fmt_expr(p.args['high'], alias_map_stack, show_table=show_table)}])"
    if isinstance(p, exp.In):
        lhs = _fmt_expr(p.this, alias_map_stack, show_table=show_table)
        vals = p.args.get("expressions")
        if vals:
            vv = ", ".join(_fmt_expr(v, alias_map_stack, show_table=show_table) for v in vals)
            return f"{kind}(IN, {lhs}, [{vv}])"
    if isinstance(p, exp.Is):
        return f"{kind}(IS, {_fmt_expr(p.left, alias_map_stack, show_table=show_table)}, {_fmt_expr(p.right, alias_map_stack, show_table=show_table)})"
    if isinstance(p, exp.Not):
        return f"{kind}(NOT, {_fmt_expr(p.this, alias_map_stack, show_table=show_table)}, TRUE)"
    return f"{kind}_TREE({p.sql()})"

def _fmt_expr(e: exp.Expression, alias_map_stack: List[Dict[str, str]], show_table: bool = True) -> str:
    if isinstance(e, exp.Column):
        table = e.table
        name = e.name
        if not show_table:
            return name.lower()  # Normalize column names to lowercase
        if table:
            return f"{_canon(alias_map_stack, table)}.{name.lower()}"  # Normalize column names to lowercase
        return name.lower()  # unqualified column, normalize to lowercase
    if isinstance(e, exp.Literal):
        if e.is_string: return f"VALUE('{e.this}')"
        return f"VALUE({e.this})"
    if isinstance(e, exp.Null):
        return "VALUE(NULL)"
    if isinstance(e, exp.Boolean):
        return f"VALUE({str(e.this).upper()})"
    if isinstance(e, exp.Paren):
        return _fmt_expr(e.this, alias_map_stack, show_table=show_table)
    if isinstance(e, exp.Func):
        return _fmt_func_or_expr(e, alias_map_stack, show_table=show_table)
    if isinstance(e, exp.Cast):
        return f"CAST({_fmt_expr(e.this, alias_map_stack, show_table=show_table)} AS {e.args.get('to') and e.args['to'].sql()})"
    if isinstance(e, exp.Tuple):
        return "(" + ", ".join(_fmt_expr(x, alias_map_stack, show_table=show_table) for x in e.expressions) + ")"
    return e.sql()

def _fmt_func_or_expr(f: exp.Func, alias_map_stack: List[Dict[str, str]], show_table: bool = True) -> str:
    # Some functions are typed nodes (e.g., exp.Max) and may not expose .name
    name = (getattr(f, "name", None) or getattr(f, "key", "") or "").upper()
    arg_nodes = []
    exprs = f.args.get("expressions")
    if exprs:
        arg_nodes.extend(exprs)
    if f.args.get("this") is not None:
        arg_nodes.append(f.args.get("this"))
    args = ", ".join(_fmt_expr(a, alias_map_stack, show_table=show_table) for a in arg_nodes)
    if name:
        if f.args.get("distinct"):
            return f"{name}(DISTINCT {args})"
        return f"{name}({args})"
    return f.sql()

def _fmt_table_and_key(t: exp.Expression) -> Tuple[str, str]:
    """
    Returns (rendered_table_name, alias_key_for_scope)
    alias_key is what we map to canonical T1/T2...
    If no alias present, we use the table name as key.
    """
    alias = None
    table_expr = t
    if isinstance(t, exp.Alias):
        alias = t.alias
        table_expr = t.this
    if isinstance(table_expr, exp.Table):
        # Render fully-qualified table (db.schema.table if present)
        ident_parts = [i.name.lower() for i in table_expr.find_all(exp.Identifier)]  # Normalize to lowercase
        rendered = ".".join(ident_parts) or table_expr.name.lower()  # Normalize to lowercase
        key = str(alias or (table_expr.alias or table_expr.name))
        return rendered, key
    # Fallback for exotic sources
    return table_expr.sql(), str(alias or "SUBQUERY")

def _canon(alias_map_stack: List[Dict[str, str]], alias_or_table: str) -> str:
    # Look up from innermost scope to outer
    for scope in reversed(alias_map_stack):
        if alias_or_table in scope:
            return scope[alias_or_table]
    # Not registered (e.g., unaliased parent reference) → keep as-is
    return alias_or_table

def _fmt_window_spec(win: exp.Window, alias_map_stack: List[Dict[str, str]]) -> str:
    parts = []
    if p := win.args.get("partition_by"):
        parts.append("partition_by=[" + ", ".join(_fmt_expr(x, alias_map_stack, show_table=False) for x in p.expressions) + "]")
    if o := win.args.get("order"):
        items = []
        for it in o.expressions:
            dir_ = "DESC" if it.args.get("desc") else "ASC"
            items.append(f"{_fmt_expr(it.this, alias_map_stack, show_table=False)} {dir_}")
        parts.append("order_by=[" + ", ".join(items) + "]")
    if f := win.args.get("frame"):
        parts.append(f"frame={f.sql()}")
    return ", ".join(parts) if parts else "default"

# ------------------------------ Demo ------------------------------------ #
if __name__ == "__main__":
    q = """
    WITH top_c AS (
      SELECT o.customer_id, SUM(o.total) s
      FROM orders o
      GROUP BY o.customer_id
      HAVING SUM(o.total) > 1000
    )
    SELECT DISTINCT c.name, t.s AS revenue
    FROM customers c
    JOIN top_c t ON t.customer_id = c.id
    WHERE c.country = 'AU' AND EXISTS (
        SELECT 1 FROM invoices i
        WHERE i.customer_id = c.id
    )
    ORDER BY revenue DESC
    LIMIT 50;
    """
    for op in sql_to_ops(q, dialect="duckdb"):
        print(op)
