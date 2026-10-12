#!/usr/bin/env python3
"""Parquet 实时查看器: FastAPI + DuckDB 后端.

用法:
    python app.py [--root /path/to/datasets] [--host 0.0.0.0] [--port 8790]

- 支持单文件浏览, 也支持目录浏览(自动 union 目录下所有 parquet 分片)
- WHERE 过滤 / 排序 / 分页 / 随机抽样 / 自定义 SQL 全部下推给 DuckDB
"""
import argparse
import json
import os
import re
import threading
import time

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.environ.get(
    "DATA_ROOT", "/hdd2/xiaolirui/octoThinker-Datasets"))
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8790)
args = parser.parse_args()

ROOT = os.path.abspath(args.root)
if not os.path.isdir(ROOT):
    raise SystemExit(f"数据目录不存在: {ROOT}")

app = FastAPI(title="Parquet Viewer")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 危险 SQL 关键字(SQL 模式 / WHERE 模式下都禁止)
BAD_SQL = re.compile(
    r"\b(copy|attach|export|import|install|load|create|drop|insert|update|"
    r"delete|alter|pragma|call|checkpoint|force|set|use|from\s+\(|read_csv|"
    r"read_text|write_file|glob)\b", re.IGNORECASE)

_scan_cache = {"files": {}, "ts": 0}
_scan_lock = threading.Lock()
SCAN_TTL = 5  # 秒; 文件树缓存, 前端轮询时不会反复扫盘

MAX_ROWS_PAGE = 2000
MAX_CELL_CHARS = 20000
MAX_SQL_ROWS = 5000


def check_fragment(sql: str):
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise HTTPException(400, "SQL 为空")
    if ";" in s:
        raise HTTPException(400, "只允许单条查询, 不允许分号拼接多条语句")
    if BAD_SQL.search(s):
        raise HTTPException(400, "检测到不允许的关键字(仅允许 SELECT 查询)")
    return s


def rel_to_abs(rel: str) -> str:
    rel = rel.lstrip("/")
    abs_path = os.path.abspath(os.path.join(ROOT, rel))
    if not (abs_path == ROOT or abs_path.startswith(ROOT + os.sep)):
        raise HTTPException(403, "路径越界")
    return abs_path


def register_view(con, abs_path: str):
    q = abs_path.replace("'", "''")
    if os.path.isdir(abs_path):
        src = f"read_parquet('{q}/*.parquet', union_by_name = true)"
    else:
        src = f"read_parquet('{q}', union_by_name = true)"
    con.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM {src}")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def cell_json(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v
    elif isinstance(v, (int, float, bool)):
        return v
    elif isinstance(v, (bytes, bytearray)):
        s = repr(bytes(v[:64])) + ("..." if len(v) > 64 else "")
    else:  # list / dict 等嵌套类型
        try:
            s = json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            s = str(v)
    if len(s) > MAX_CELL_CHARS:
        s = s[:MAX_CELL_CHARS] + f" …[已截断, 原长度 {len(s)} 字符]"
    return s


def run_rows(con, sql: str, cap: int):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()[:cap]
    return {"columns": cols, "rows": [[cell_json(v) for v in r] for r in rows],
            "truncated": len(rows) == cap}


def scan_files(force=False):
    with _scan_lock:
        now = time.time()
        if not force and now - _scan_cache["ts"] < SCAN_TTL:
            return _scan_cache["files"]
        files = {}
        for dirpath, _dirnames, filenames in os.walk(ROOT):
            for fn in filenames:
                if not fn.endswith(".parquet"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, ROOT)
                files[rel] = {"size": st.st_size, "mtime": st.st_mtime}
        _scan_cache["files"] = files
        _scan_cache["ts"] = now
        return files


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/scan")
def api_scan(force: bool = False):
    files = scan_files(force=force)
    dirs = {}
    for rel, meta in files.items():
        parts = rel.split(os.sep)
        node = dirs
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = meta
    return {"root": ROOT, "tree": dirs, "count": len(files)}


@app.get("/api/open")
def api_open(path: str = Query(...)):
    abs_path = rel_to_abs(path)
    if not (os.path.isfile(abs_path) or os.path.isdir(abs_path)):
        raise HTTPException(404, f"不存在: {path}")
    con = duckdb.connect()
    try:
        register_view(con, abs_path)
        cols = [{"name": r[0], "type": r[1]}
                for r in con.execute("DESCRIBE SELECT * FROM data").fetchall()]
        total = con.execute("SELECT count(*) FROM data").fetchone()[0]
        return {"path": path, "is_dir": os.path.isdir(abs_path),
                "columns": cols, "total": total}
    finally:
        con.close()


class QueryReq(BaseModel):
    path: str
    where: str = ""
    order_by: str = ""
    order_dir: str = "asc"
    page: int = 1
    page_size: int = 100
    sample: bool = False


@app.post("/api/query")
def api_query(req: QueryReq):
    abs_path = rel_to_abs(req.path)
    con = duckdb.connect()
    try:
        register_view(con, abs_path)
        where = ""
        if req.where.strip():
            where = " WHERE " + check_fragment(req.where)
        total = con.execute(f"SELECT count(*){'' if where else ''} FROM data{where}"
                            ).fetchone()[0]
        order = ""
        if req.order_by:
            direction = "DESC" if req.order_dir.lower() == "desc" else "ASC"
            order = f" ORDER BY {qident(req.order_by)} {direction}"
        elif req.sample:
            order = " ORDER BY random()"
        page_size = min(max(req.page_size, 1), MAX_ROWS_PAGE)
        page = max(req.page, 1)
        sql = (f"SELECT * FROM data{where}{order} "
               f"LIMIT {page_size} OFFSET {(page - 1) * page_size}")
        out = run_rows(con, sql, MAX_ROWS_PAGE)
        out.update(total=total, page=page, page_size=page_size)
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"查询失败: {e}")
    finally:
        con.close()


class SqlReq(BaseModel):
    path: str
    sql: str


@app.post("/api/sql")
def api_sql(req: SqlReq):
    abs_path = rel_to_abs(req.path)
    sql = check_fragment(req.sql)
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise HTTPException(400, "只允许 SELECT / WITH 查询")
    con = duckdb.connect()
    try:
        register_view(con, abs_path)
        return run_rows(con, sql, MAX_SQL_ROWS)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"查询失败: {e}")
    finally:
        con.close()


@app.get("/api/profile")
def api_profile(path: str = Query(...), column: str = Query(...)):
    abs_path = rel_to_abs(path)
    con = duckdb.connect()
    try:
        register_view(con, abs_path)
        col = qident(column)
        ctype = ""
        for r in con.execute("DESCRIBE SELECT * FROM data").fetchall():
            if r[0] == column:
                ctype = r[1]
                break
        numeric = bool(re.match(r"^(tinyint|smallint|integer|bigint|hugeint|"
                                r"float|double|decimal|ubigint|usmallint)", ctype))
        exprs = [f"count(*) AS total",
                 f"count({col}) AS non_null",
                 f"min({col}) AS min", f"max({col}) AS max",
                 f"approx_count_distinct({col}) AS approx_distinct"]
        if numeric:
            exprs.append(f"avg({col}) AS avg")
        row = con.execute("SELECT " + ", ".join(exprs) + " FROM data").fetchone()
        keys = ["total", "non_null", "min", "max", "approx_distinct"]
        if numeric:
            keys.append("avg")
        return {"column": column, "type": ctype,
                "stats": {k: cell_json(v) for k, v in zip(keys, row)}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"统计失败: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    import uvicorn
    print(f"数据根目录: {ROOT}")
    print(f"打开浏览器访问: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
