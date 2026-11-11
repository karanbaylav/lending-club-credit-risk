# Assumptions and choices

Every number or rule chosen by me rather than estimated from the data goes here, with the reason.
Rules marked **fixed before results** were written before any model output was looked at.

## Data

- Source: Lending Club accepted loans 2007–2018Q4 (Kaggle, `wordsforthewise/lending-club`), FRED `UNRATE` and `TB3MS`. Licence and download date: see README.
- Snapshot date = the latest payment month in the file. Ages "at the snapshot" are measured to that month.
- Rows without a loan amount are file footers and are dropped.

## Default definition (fixed before results)

- **Default** = loan status `Charged Off`, `Default`, or `Does not meet the credit policy. Status:Charged Off`.
- **Non-default (resolved)** = `Fully Paid` or `Does not meet the credit policy. Status:Fully Paid`.
- **Open** = everything else (`Current`, `In Grace Period`, `Late (16–30 days)`, `Late (31–120 days)`): outcome unknown.
- "Does not meet the credit policy" loans (2007–2010, issued outside the later credit policy) are kept with their underlying outcome and flagged `policy_fail`, because they are real loans with real outcomes. Sensitivity: results are re-run without them where it matters.
- **Default month** = the month after the last payment received (the first missed payment), or month 1 if nothing was ever paid. Lending Club charges a loan off months later; our event date is the first missed payment of a loan that ended in default.

## Two targets

- **Lifetime outcome** (Stages 1–2): default vs fully paid, on resolved loans only. Open loans are dropped, never counted as non-default. For recent issue years this sample over-represents defaults and early payoffs (right-censoring); every recent-year number carries that warning.
- **12-month default** (Stage 3 PD model): default month ≤ 12. Loans that are paid off within 12 months count as **non-defaults** (standard practice: they survived the window). Loans that default after month 12 also count as non-defaults for this target. This outcome is known for any loan at least **18 months old at the snapshot** (12 months plus about 6 months for a missed payment to show up as a charge-off). Open loans that old count as non-defaults at 12 months, which avoids the censoring bias of the lifetime target.

## Time split for the PD model (fixed before results)

- Train: issued 2007–2014.
- Validation: issued 2015.
- Test: issued 2016 onward, and at least 18 months before the snapshot.
- Rows are never shuffled across years. Binning, WoE, hyperparameters and any cutoff are chosen on train and validation only.

## Feature treatment

- Allowed features: origination-time fields only (`loan_features` table). Payments, recoveries, last payment date, outstanding principal, later FICO pulls, hardship and settlement fields are excluded from all models.
- Interest rate, grade and sub-grade are Lending Club's own risk pricing. The scorecard is built **without** them (borrower information only), so it can be compared with Lending Club's grades; a version with grade is reported separately.
- Income ≤ 0, debt-to-income < 0 or > 100, and FICO outside 300–850 are set to missing. Revolving utilisation is capped at 150%.

## Choices made after review (logged, not hidden)

- **Mortgage accounts (`mort_acc`) dropped from the models.** The field was not recorded before 2012 (100% missing for 2007–2011), so its "missing" bin disappears after the training years and its PSI is 1.29: a recording change, not borrower drift. With it, the scorecard's test Gini is 0.2986; without it, 0.2980. Kept as a sensitivity check in `reports/tables/s3_sensitivity_mort_acc.csv`.
- **Gradient boosting benchmark tuned lightly.** Three regularisation settings, each early-stopped on 2015; the pick uses 2015 Gini only. Test Gini is reported for the chosen setting only.
- **Intercept shift (diagnostic only).** A log-odds shift is fitted on 2015 and applied unchanged to 2016–17 to show how much of the calibration gap a simple update closes. The model is never recalibrated on the test years.
- **Realised return sample** includes 60-month loans issued up to 2013; the same tables are repeated for 36-month loans only, to separate term from pricing.

## Seeds

- Random seed `20261111` everywhere (bootstrap, sampling, gradient boosting).
