-- =====================================================================
-- Credit risk | 01 — Load raw files into DuckDB (all columns as text; typed in 02)
-- Sources (data/raw, not committed): Lending Club accepted loans 2007–2018Q4 (Kaggle,
-- wordsforthewise/lending-club), FRED UNRATE and TB3MS (monthly).
-- =====================================================================
CREATE OR REPLACE TABLE raw_accepted AS
SELECT * FROM read_csv('{ACCEPTED}',
                       header = true, all_varchar = true, quote = '"', escape = '"',
                       max_line_size = 10000000, ignore_errors = false);

-- @load_counts  (rows and columns as loaded; footer rows have no loan amount)
SELECT (SELECT COUNT(*) FROM raw_accepted)                                   AS rows_loaded,
       (SELECT COUNT(*) FROM raw_accepted WHERE loan_amnt IS NULL)           AS rows_without_loan_amount,
       (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'raw_accepted') AS columns_loaded;

-- @load_missing  (missing values per raw column, saved before any cleaning)
WITH s AS (SELECT * FROM (SUMMARIZE raw_accepted))
SELECT column_name, null_percentage, approx_unique FROM s ORDER BY column_name;
