"""Shared paths and the DuckDB connection (memory-capped so it runs on a laptop)."""
import os
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
# set LC_DB to keep the database elsewhere (e.g. a faster disk); default is data/clean/
DB = pathlib.Path(os.environ.get("LC_DB", ROOT / "data" / "clean" / "lc.duckdb"))
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures"
SEED = 20261111


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=read_only)
    tmp = DB.parent / "tmp_duckdb"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    return con


def q(con, sql: str):
    """Run a query and return a DataFrame."""
    return con.execute(sql).df()
