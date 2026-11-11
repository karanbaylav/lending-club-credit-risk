"""Run numbered SQL scripts in order: python -m src.run sql/01_load.sql sql/02_clean.sql ...

Statements are split on a semicolon at the end of a line. A statement preceded by a
`-- @name description` tag is a report query: its result is saved to
reports/tables/<name>.csv. A tag starting with `-- @check` must return a column
`failures`; the run stops if any failure count is above zero."""
import os
import re
import sys
import time

from src.db import ROOT, TABLES, connect


def statements(sql: str):
    for chunk in re.split(r";[ \t]*\n", sql + "\n"):
        body = re.sub(r"--[^\n]*", "", chunk).strip()
        if not body:
            continue
        tag = re.search(r"^--\s*@(\w+)", chunk, flags=re.M)
        yield (tag.group(1) if tag else None), chunk


def run_file(con, path, save=True):
    t0 = time.time()
    sql = (ROOT / path).read_text()
    raw = ROOT / "data" / "raw"
    sql = sql.replace("{RAW}", str(raw))
    for stem in ("accepted", "rejected"):
        hits = sorted(raw.glob(f"{stem}_2007_to_2018Q4.csv*"))
        if hits:
            sql = sql.replace("{" + stem.upper() + "}", str(hits[0]))
    for name, stmt in statements(sql):
        if name and not save:
            continue
        if name:
            df = con.execute(stmt).df()
            TABLES.mkdir(parents=True, exist_ok=True)
            df.to_csv(TABLES / f"{name}.csv", index=False)
            if name.startswith("check") and "failures" in df and (df["failures"] > 0).any():
                print(df.to_string(index=False))
                sys.exit(f"Blocking check failed in {path}: {name}")
        else:
            con.execute(stmt)
    print(f"{path}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    con = connect()
    for p in sys.argv[1:]:
        run_file(con, p)
    con.close()
