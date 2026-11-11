-- =====================================================================
-- Credit risk | 04 — Vintage curves (Stage 2)
-- Cumulative default rate by loan age for each issue-year vintage and grade, 36-month
-- loans. Every loan in the vintage (open, paid or defaulted) stays in the denominator, so
-- prepaid loans count as "did not default". Ages are shown only up to the youngest loan's
-- age at the snapshot minus 6 months, so defaults that have not yet been charged off
-- cannot pull the curve down.
-- =====================================================================
CREATE OR REPLACE TABLE vintage_curves AS
WITH base AS (
    SELECT year(e.issue_month) AS vintage, f.grade, e.is_default, e.exit_age, e.age_at_snapshot
    FROM loan_events e JOIN loan_features f USING (loan_id)
    WHERE e.term_months = 36 AND f.grade IS NOT NULL
), cohort AS (
    SELECT vintage, grade, COUNT(*) AS loans, MIN(age_at_snapshot) - 6 AS max_age
    FROM base GROUP BY ALL
), ages AS (SELECT * FROM range(1, 37) a(age)),
defaults_by_age AS (
    SELECT vintage, grade, exit_age AS age, COUNT(*) AS new_defaults
    FROM base WHERE is_default GROUP BY ALL
)
SELECT c.vintage, c.grade, a.age, c.loans,
       SUM(COALESCE(d.new_defaults, 0)) OVER (PARTITION BY c.vintage, c.grade ORDER BY a.age) AS cum_defaults,
       100.0 * SUM(COALESCE(d.new_defaults, 0)) OVER (PARTITION BY c.vintage, c.grade ORDER BY a.age) / c.loans AS cum_default_pct
FROM cohort c CROSS JOIN ages a
LEFT JOIN defaults_by_age d ON d.vintage = c.vintage AND d.grade = c.grade AND d.age = a.age
QUALIFY a.age <= c.max_age;

-- @vintage_curves  (one row per vintage x grade x age, 36-month loans)
SELECT vintage, grade, age, loans, cum_defaults, ROUND(cum_default_pct, 3) AS cum_default_pct
FROM vintage_curves ORDER BY grade, vintage, age;

-- @vintage_at_12_24_36  (cumulative default % at fixed ages; blank = not yet observable)
SELECT grade, vintage, MAX(loans) AS loans,
       ROUND(MAX(cum_default_pct) FILTER (WHERE age = 12), 2) AS pct_by_12m,
       ROUND(MAX(cum_default_pct) FILTER (WHERE age = 24), 2) AS pct_by_24m,
       ROUND(MAX(cum_default_pct) FILTER (WHERE age = 36), 2) AS pct_by_36m
FROM vintage_curves GROUP BY ALL ORDER BY grade, vintage;
