# make all rebuilds every table, figure and report from data/raw
PY ?= python

.PHONY: all stage1 stage2 stage3 test clean

all: stage1 stage2 stage3 test

stage1:
	rm -rf data/clean/gbm_cache
	$(PY) -m src.run sql/01_load.sql sql/02_clean.sql sql/03_quality.sql
	$(PY) -m src.stage1_report

stage2:
	$(PY) -m src.run sql/04_vintage.sql
	$(PY) -m src.stage2_returns

stage3:
	$(PY) -m src.stage3_scorecard

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf data/clean/lc.duckdb data/clean/gbm_cache
