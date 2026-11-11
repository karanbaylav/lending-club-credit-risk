-- =====================================================================
-- Credit risk | 03 — Data-quality checks (blocking checks stop the build)
-- =====================================================================

-- @check_keys  (blocking: must all be 0)
SELECT 'duplicate loan ids' AS test, COUNT(*) - COUNT(DISTINCT loan_id) AS failures FROM loans_typed
UNION ALL SELECT 'missing loan id', COUNT(*) FILTER (WHERE loan_id IS NULL) FROM loans_typed
UNION ALL SELECT 'missing or unparsed issue month', COUNT(*) FILTER (WHERE issue_month IS NULL) FROM loans_typed
UNION ALL SELECT 'missing grade', COUNT(*) FILTER (WHERE grade IS NULL) FROM loans_typed
UNION ALL SELECT 'term not 36 or 60 months', COUNT(*) FILTER (WHERE term_months NOT IN (36, 60) OR term_months IS NULL) FROM loans_typed;

-- @quality_values  (impossible or suspicious values; counted, then treated in src/features.py)
SELECT 'annual income <= 0' AS issue, COUNT(*) FILTER (WHERE annual_inc <= 0) AS loans, 'set to missing' AS treatment FROM loans_typed
UNION ALL SELECT 'annual income > 5,000,000', COUNT(*) FILTER (WHERE annual_inc > 5e6), 'kept (WoE bins and trees are insensitive to extremes)' FROM loans_typed
UNION ALL SELECT 'debt-to-income < 0', COUNT(*) FILTER (WHERE dti < 0), 'set to missing' FROM loans_typed
UNION ALL SELECT 'debt-to-income > 100', COUNT(*) FILTER (WHERE dti > 100), 'set to missing (sentinel or data error)' FROM loans_typed
UNION ALL SELECT 'revolving utilisation > 150%', COUNT(*) FILTER (WHERE revol_util > 150), 'capped at 150' FROM loans_typed
UNION ALL SELECT 'FICO outside 300-850', COUNT(*) FILTER (WHERE fico < 300 OR fico > 850), 'set to missing' FROM loans_typed
UNION ALL SELECT 'funded amount > requested amount', COUNT(*) FILTER (WHERE funded_amnt > loan_amnt), 'kept' FROM loans_typed
UNION ALL SELECT 'last payment before issue', COUNT(*) FILTER (WHERE last_pymnt_month < issue_month), 'dates out of order; exit age floored at 1' FROM loans_typed
UNION ALL SELECT 'first credit line after issue', COUNT(*) FILTER (WHERE earliest_cr_line > issue_month), 'history set to missing' FROM loans_typed
UNION ALL SELECT 'principal or interest received but no last-payment date', COUNT(*) FILTER (WHERE last_pymnt_month IS NULL AND total_rec_prncp + total_rec_int > 0), 'exit age set to 1' FROM loans_typed
UNION ALL SELECT 'interest rate outside 5%-31%', COUNT(*) FILTER (WHERE int_rate < 5 OR int_rate > 31), 'kept' FROM loans_typed;

-- @missing_features  (share missing per origination-time feature)
WITH s AS (SELECT * FROM (SUMMARIZE loan_features))
SELECT column_name AS feature, column_type, ROUND(CAST(null_percentage AS DOUBLE), 2) AS pct_missing
FROM s ORDER BY pct_missing DESC, feature;

-- @default_rate_by_year  (first drift plot; the resolved-only rate is censored for recent years)
SELECT year(e.issue_month) AS issue_year,
       COUNT(*)                                                         AS loans,
       ROUND(100.0 * AVG((e.status_group = 'open')::INT), 1)            AS pct_still_open,
       COUNT(*) FILTER (WHERE e.status_group <> 'open')                 AS resolved_loans,
       ROUND(100.0 * AVG(e.is_default::INT) FILTER (WHERE e.status_group <> 'open'), 2) AS default_rate_resolved_pct,
       COUNT(*) FILTER (WHERE e.obs_12m)                                AS loans_12m_known,
       ROUND(100.0 * AVG(e.bad_12m::INT) FILTER (WHERE e.obs_12m), 2)   AS default_rate_12m_pct,
       ROUND(100.0 * AVG((e.term_months = 60)::INT), 1)                 AS pct_60_month
FROM loan_events e GROUP BY 1 ORDER BY 1;
