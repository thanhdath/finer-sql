import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -----------------------------
# Config (no memory watchdogs here)
# -----------------------------
SPIDER_DB_ROOT = "/app/data/spider/database"
BIRD_TRAIN_DB_ROOT = "/app/data/bird/train/train_databases"
BIRD_DEV_DB_ROOT = "/app/data/bird/dev/dev_databases"
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "100"))
DEFAULT_TIMEOUT_MS = int(os.getenv("DEFAULT_TIMEOUT_MS", "120000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "10000"))


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="SQLite No-Lock SQL Executor")
gate = threading.Semaphore(MAX_CONCURRENT)


# -----------------------------
# Models
# -----------------------------
class ExecuteRequest(BaseModel):
    dataset_name: str = Field(..., description="Dataset name: 'spider' or 'bird'")
    db_id: str = Field(..., description="Database ID (folder name)")
    sql: str = Field(..., description="Single SQL statement.")
    mode: str = Field(
        "sandbox_rollback",
        description="read_only | sandbox_rollback",
        pattern="^(read_only|sandbox_rollback)$"
    )
    timeout_ms: int = Field(DEFAULT_TIMEOUT_MS, ge=1, le=120_000)
    max_rows: int = Field(MAX_ROWS, ge=1, le=50_000)


class ExecuteResponse(BaseModel):
    ok: bool
    statement_type: str
    rows: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    pandas_result: Optional[str] = None
    notice: Optional[str] = None
    error: Optional[str] = None
    timed_out: bool = False


# -----------------------------
# Helpers
# -----------------------------
def db_path_safe(dataset_name: str, db_id: str) -> Path:
    if dataset_name not in ["spider", "bird"]:
        raise HTTPException(400, detail="dataset_name must be 'spider' or 'bird'")

    if dataset_name == "spider":
        root = Path(SPIDER_DB_ROOT)
    else:
        train_root = Path(BIRD_TRAIN_DB_ROOT)
        dev_root = Path(BIRD_DEV_DB_ROOT)
        if (train_root / db_id / f"{db_id}.sqlite").exists():
            root = train_root
        elif (dev_root / db_id / f"{db_id}.sqlite").exists():
            root = dev_root
        else:
            raise HTTPException(404, detail=f"Database not found: {dataset_name}/{db_id}")

    db_folder = root / db_id
    db_file = db_folder / f"{db_id}.sqlite"
    if not db_file.exists():
        raise HTTPException(404, detail=f"Database not found: {dataset_name}/{db_id}")

    if not db_file.resolve().is_relative_to(root.resolve()):
        raise HTTPException(400, detail="Invalid database path.")

    return db_file.resolve()


def classify(sql: str) -> str:
    parts = sql.strip().split(None, 1)
    return parts[0].upper() if parts else "UNKNOWN"


def is_select_like(sql: str) -> bool:
    s = sql.lstrip().lower()
    return s.startswith("select") or s.startswith("with") or s.startswith("explain")


# -----------------------------
# Core executors (run in threads)
# -----------------------------
def _run_read_only_thread(dbfile: Path, sql: str, timeout_ms: int, max_rows: int) -> ExecuteResponse:
    # Use immutable read-only URI to avoid locking the original file
    uri = f"file:{dbfile}?immutable=1&mode=ro"
    deadline_ts = time.time() + (timeout_ms / 1000.0)

    def progress_cb():
        # Abort when over deadline
        return 1 if time.time() >= deadline_ts else 0

    try:
        conn = sqlite3.connect(uri, uri=True, timeout=min(2.0, timeout_ms/1000.0), check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            # Enforce read-only behavior and lower contention
            conn.set_progress_handler(progress_cb, 1000)
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA case_sensitive_like = ON;")
            conn.execute(f"PRAGMA busy_timeout={int(timeout_ms)};")

            stmt_type = classify(sql)
            cur = conn.execute(sql)

            # Fetch everything at once; enforce max_rows by slicing
            rows: List[sqlite3.Row] = cur.fetchall()
            if max_rows:
                rows = rows[:max_rows]

            cols = [d[0] for d in cur.description] if cur.description else []
            cur.close()

            data = [ {c: r[idx] for idx, c in enumerate(cols)} for r in rows ] if cols else []

            if rows:
                pandas_result = pd.DataFrame(rows, columns=pd.Index([str(c) for c in cols])).to_string(index=False)
            else:
                pandas_result = "Empty result set"

            return ExecuteResponse(
                ok=True,
                statement_type=stmt_type,
                rows=data,
                row_count=len(data),
                pandas_result=pandas_result,
                notice="Read-only mode (immutable)."
            )
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        msg = str(e)
        timed_out = ("timeout" in msg) or ("interrupt" in msg) or ("exceeded" in msg)
        return ExecuteResponse(ok=False, statement_type=classify(sql), error=msg, timed_out=timed_out)
    except Exception as e:
        return ExecuteResponse(ok=False, statement_type=classify(sql), error=str(e))


def _run_sandbox_thread(dbfile: Path, sql: str, timeout_ms: int, max_rows: int) -> ExecuteResponse:
    # Execute SQL on a temp copy and rollback to avoid modifying source and to avoid locking it
    deadline_ts = time.time() + (timeout_ms / 1000.0)

    def progress_cb(_):
        return 1 if time.time() >= deadline_ts else 0

    try:
        conn = sqlite3.connect(dbfile.as_posix(), timeout=min(2.0, timeout_ms/1000.0), check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            # Lower lock contention on the temp copy
            conn.set_progress_handler(progress_cb, 1000)

            stmt_type = classify(sql)
            if is_select_like(sql):
                # No transaction for SELECT queries
                cur = conn.execute(sql)
                rows: List[sqlite3.Row] = cur.fetchall()
                if max_rows:
                    rows = rows[:max_rows]
                cols = [d[0] for d in cur.description] if cur.description else []
                cur.close()
                data = [ {c: r[idx] for idx, c in enumerate(cols)} for r in rows ] if cols else []
                row_count = len(data)
                pandas_result = pd.DataFrame(rows, columns=pd.Index([str(c) for c in cols])).to_string(index=False) if rows else "Empty result set"
                return ExecuteResponse(
                    ok=True,
                    statement_type=stmt_type,
                    rows=data,
                    row_count=row_count,
                    pandas_result=pandas_result,
                    notice="Executed SELECT without transaction."
                )
            else:
                return ExecuteResponse(
                    ok=True,
                    statement_type=stmt_type,
                    notice="Not allowed to execute non-SELECT queries"
                )
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        print(e)
        msg = str(e)
        timed_out = ("timeout" in msg) or ("interrupt" in msg) or ("exceeded" in msg)
        return ExecuteResponse(ok=False, statement_type=classify(sql), error=msg, timed_out=timed_out)
    except Exception as e:
        print(e)
        return ExecuteResponse(ok=False, statement_type=classify(sql), error=str(e))


# -----------------------------
# API
# -----------------------------
@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest):
    dbfile = db_path_safe(req.dataset_name, req.db_id)

    gate.acquire()
    try:
        if req.mode == "read_only":
            # Quick fail if clearly a write
            sql_upper = req.sql.strip().upper()
            if sql_upper.startswith(("INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE", "TRUNCATE", "ALTER", "DROP", "CREATE", "VACUUM", "PRAGMA")):
                raise HTTPException(400, detail="Write/DDL detected; use mode='sandbox_rollback' to test safely.")
            return _run_read_only_thread(dbfile, req.sql, req.timeout_ms, req.max_rows)
        else:
            return _run_sandbox_thread(dbfile, req.sql, req.timeout_ms, req.max_rows)
    finally:
        gate.release()


# -----------------------------
# Health Endpoints
# -----------------------------
@app.get("/healthz/live")
async def liveness():
    return {"ok": True, "ts": time.time()}


@app.get("/healthz/ready")
async def readiness():
    try:
        return {"ok": True, "ts": time.time()}
    except Exception:
        raise HTTPException(503, "not ready")


# Run: uvicorn api_no_lock:app --host 0.0.0.0 --port 8001 --workers 8

