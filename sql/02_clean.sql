-- =====================================================================
-- Credit risk | 02 — Typed loan table, outcomes, observation windows
-- Origination-time features and post-origination (outcome) fields are kept apart:
--   loan_features : what was known when the loan was issued (models may use these)
--   loan_outcomes : what happened afterwards (targets and return analysis only)
-- Rules: ASSUMPTIONS.md (default definition, 12-month target, snapshot, time split).
-- =====================================================================

-- 1. Typed base table (footer rows without a loan amount are dropped)
CREATE OR REPLACE TABLE loans_typed AS
SELECT
    TRY_CAST(id AS BIGINT)                                           AS loan_id,
    TRY_CAST(loan_amnt AS DOUBLE)                                    AS loan_amnt,
    TRY_CAST(funded_amnt AS DOUBLE)                                  AS funded_amnt,
    TRY_CAST(regexp_extract(term, '(\d+)', 1) AS INTEGER)            AS term_months,
    TRY_CAST(replace(int_rate, '%', '') AS DOUBLE)                   AS int_rate,
    TRY_CAST(installment AS DOUBLE)                                  AS installment,
    grade, sub_grade,
    CASE WHEN emp_length LIKE '< 1%' THEN 0
         WHEN emp_length LIKE '10+%' THEN 10
         ELSE TRY_CAST(regexp_extract(emp_length, '(\d+)', 1) AS INTEGER) END AS emp_length_yrs,
    home_ownership,
    TRY_CAST(annual_inc AS DOUBLE)                                   AS annual_inc,
    verification_status,
    TRY_CAST(strptime('01-' || issue_d, '%d-%b-%Y') AS DATE)         AS issue_month,
    loan_status, purpose, addr_state,
    TRY_CAST(dti AS DOUBLE)                                          AS dti,
    TRY_CAST(delinq_2yrs AS DOUBLE)                                  AS delinq_2yrs,
    TRY_CAST(strptime('01-' || earliest_cr_line, '%d-%b-%Y') AS DATE) AS earliest_cr_line,
    (TRY_CAST(fico_range_low AS DOUBLE) + TRY_CAST(fico_range_high AS DOUBLE)) / 2 AS fico,
    TRY_CAST(inq_last_6mths AS DOUBLE)                               AS inq_last_6mths,
    TRY_CAST(open_acc AS DOUBLE)                                     AS open_acc,
    TRY_CAST(pub_rec AS DOUBLE)                                      AS pub_rec,
    TRY_CAST(revol_bal AS DOUBLE)                                    AS revol_bal,
    TRY_CAST(replace(revol_util, '%', '') AS DOUBLE)                 AS revol_util,
    TRY_CAST(total_acc AS DOUBLE)                                    AS total_acc,
    TRY_CAST(mort_acc AS DOUBLE)                                     AS mort_acc,
    TRY_CAST(pub_rec_bankruptcies AS DOUBLE)                         AS pub_rec_bankruptcies,
    application_type, initial_list_status,
    TRY_CAST(policy_code AS DOUBLE)                                  AS policy_code,
    -- post-origination fields (never model inputs)
    TRY_CAST(total_pymnt AS DOUBLE)                                  AS total_pymnt,
    TRY_CAST(total_rec_prncp AS DOUBLE)                              AS total_rec_prncp,
    TRY_CAST(total_rec_int AS DOUBLE)                                AS total_rec_int,
    TRY_CAST(total_rec_late_fee AS DOUBLE)                           AS total_rec_late_fee,
    TRY_CAST(recoveries AS DOUBLE)                                   AS recoveries,
    TRY_CAST(collection_recovery_fee AS DOUBLE)                      AS collection_recovery_fee,
    TRY_CAST(out_prncp AS DOUBLE)                                    AS out_prncp,
    TRY_CAST(strptime('01-' || last_pymnt_d, '%d-%b-%Y') AS DATE)    AS last_pymnt_month
FROM raw_accepted
WHERE loan_amnt IS NOT NULL;

-- 2. Data snapshot = latest payment month in the file
CREATE OR REPLACE TABLE snapshot AS
SELECT MAX(last_pymnt_month) AS snapshot_month FROM loans_typed;

-- 3. Outcomes. Status groups: default (Charged Off, Default, and the "does not meet the
--    credit policy" charged-off loans), paid (Fully Paid and its policy twin), open (the rest).
--    Default month = the month after the last payment (first missed payment), or month 1
--    if nothing was ever paid. This is our event date; Lending Club's charge-off comes later.
CREATE OR REPLACE TABLE loan_outcomes AS
SELECT t.loan_id, t.issue_month, t.term_months, t.funded_amnt, t.int_rate, t.installment,
       t.loan_status,
       CASE WHEN t.loan_status IN ('Charged Off', 'Default',
                                   'Does not meet the credit policy. Status:Charged Off') THEN 'default'
            WHEN t.loan_status IN ('Fully Paid',
                                   'Does not meet the credit policy. Status:Fully Paid') THEN 'paid'
            ELSE 'open' END                                                        AS status_group,
       t.loan_status LIKE 'Does not meet%'                                         AS policy_fail,
       date_diff('month', t.issue_month, s.snapshot_month)                         AS age_at_snapshot,
       CASE WHEN t.last_pymnt_month IS NULL THEN 1
            ELSE GREATEST(1, date_diff('month', t.issue_month, t.last_pymnt_month) + 1) END AS default_age_if_default,
       GREATEST(0, date_diff('month', t.issue_month, t.last_pymnt_month))         AS last_pymnt_age,
       t.total_pymnt, t.total_rec_prncp, t.total_rec_int, t.total_rec_late_fee,
       t.recoveries, t.collection_recovery_fee, t.out_prncp, t.last_pymnt_month
FROM loans_typed t, snapshot s;

-- exit age: month of default, payoff, or the snapshot for open loans (Stage 2 and 4 use it)
CREATE OR REPLACE TABLE loan_events AS
SELECT loan_id, issue_month, term_months, status_group, policy_fail, age_at_snapshot,
       CASE status_group WHEN 'default' THEN default_age_if_default
                         WHEN 'paid'    THEN GREATEST(1, last_pymnt_age)
                         ELSE age_at_snapshot END                                 AS exit_age,
       status_group = 'default'                                                   AS is_default,
       -- 12-month target: default event within 12 months of issue
       (status_group = 'default' AND default_age_if_default <= 12)                AS bad_12m,
       -- the 12-month outcome is known once the loan is 18+ months old at the snapshot
       -- (12 months plus time for a missed payment to reach charge-off)
       age_at_snapshot >= 18                                                      AS obs_12m
FROM loan_outcomes;

-- 4. Origination-time features only (the allowed-feature list for every model)
CREATE OR REPLACE TABLE loan_features AS
SELECT loan_id, issue_month,
       year(issue_month) AS issue_year,
       year(issue_month) || 'Q' || quarter(issue_month) AS issue_quarter,
       loan_amnt, term_months, int_rate, installment, grade, sub_grade,
       emp_length_yrs, home_ownership, annual_inc, verification_status, purpose, addr_state,
       dti, delinq_2yrs, date_diff('month', earliest_cr_line, issue_month) AS credit_history_months,
       fico, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, mort_acc,
       pub_rec_bankruptcies, application_type, initial_list_status,
       12 * installment / NULLIF(annual_inc, 0) AS installment_to_income
FROM loans_typed;

-- 5. Stage 1 clean table: fully observed (resolved) loans with a final outcome
CREATE OR REPLACE TABLE loans_resolved AS
SELECT f.*, e.status_group, e.is_default, e.exit_age, e.policy_fail
FROM loan_features f JOIN loan_events e USING (loan_id, issue_month)
WHERE e.status_group IN ('default', 'paid');

-- @row_funnel  (what each rule removes)
SELECT 'rows in file' AS step, (SELECT COUNT(*) FROM raw_accepted) AS loans
UNION ALL SELECT 'summary rows without a loan amount (dropped)', (SELECT COUNT(*) FROM raw_accepted WHERE loan_amnt IS NULL)
UNION ALL SELECT 'loans', (SELECT COUNT(*) FROM loans_typed)
UNION ALL SELECT '... still open at the snapshot (no final outcome)', (SELECT COUNT(*) FROM loan_events WHERE status_group = 'open')
UNION ALL SELECT '... resolved: paid', (SELECT COUNT(*) FROM loan_events WHERE status_group = 'paid')
UNION ALL SELECT '... resolved: default', (SELECT COUNT(*) FROM loan_events WHERE status_group = 'default')
UNION ALL SELECT '... of which "does not meet the credit policy" (kept, flagged)', (SELECT COUNT(*) FROM loan_events WHERE policy_fail)
UNION ALL SELECT 'loans with a known 12-month outcome (18+ months old)', (SELECT COUNT(*) FROM loan_events WHERE obs_12m);

-- @status_map  (every status in the file and where it goes)
SELECT loan_status, status_group, COUNT(*) AS loans
FROM loan_outcomes GROUP BY ALL ORDER BY loans DESC;

-- @snapshot
SELECT snapshot_month, (SELECT MIN(issue_month) FROM loans_typed) AS first_issue,
       (SELECT MAX(issue_month) FROM loans_typed) AS last_issue
FROM snapshot;
