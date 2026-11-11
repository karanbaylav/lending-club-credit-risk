# Credit Risk Project Playbook: Lifetime Credit Loss Model

Nov 11, 2026 · @karan

## Goal and outcome

You will build a lifetime credit-loss model on Lending Club loans and use it to answer one question: did the platform price risk correctly, and what approval policy would have earned more? Plan 4 to 6 weeks at about 10 hours a week. A usable version exists after two weekends.

The project has three versions, each finished and publishable on its own. Stop at any of them.

| Version | Stages | Time | What it shows |
| --- | --- | --- | --- |
| v1: Core | 1, 2, 3 | 2 weekends | Clean data, vintage and pricing analysis, a validated PD scorecard |
| v2: Lifetime | 4, 5 | +2 weeks | Survival model, macro variables, scenario-weighted ECL |
| v3: Policy and model risk | 6, 7 | +1 to 2 weeks | Profit-based cutoff, backtest, reject inference, model-risk limits |

**What you can claim.** Claim only what you finished. After v1: "Built and validated a PD scorecard on Lending Club loans, with vintage and pricing analysis." After v2: add "lifetime PD with macro covariates and scenario-weighted IFRS 9-style ECL." After v3: add "profit-based approval policy and model-risk backtest."

**What not to claim.** Do not write regulatory-grade modelling, IRB or any banking experience. LGD, scenario paths and staging rules are your own stated assumptions. Do not state findings in the README until you have them; if the macro variables turn out insignificant, that is a finding and you report it.

**Verify first.** I am recalling the dataset layout from memory. Before starting, check the Kaggle page for the licence, the file names and the column list, and adjust the plan to what is actually there.

## Setup (about 2 hours)

**Tools.** Python 3.11+, DuckDB, pandas, scikit-learn, statsmodels, lifelines (survival models), shap, matplotlib, pytest. Use a virtual environment and a `requirements.txt` with pinned versions.

**Data.** Download three sources, and write the licence and download date in the README.

| Source | What it holds | Use |
| --- | --- | --- |
| Lending Club accepted loans (Kaggle) | One row per loan: amount, rate, grade, income, debt-to-income, issue date, final status, payments, recoveries | Everything |
| Lending Club rejected loans (Kaggle) | Rejected applications with few fields: amount, score, debt-to-income, state, date | Stage 7 only; may be too thin |
| FRED (St. Louis Fed) | US unemployment rate and an interest-rate series, monthly | Stage 4 onward |

The column names above are from memory; confirm them. The accepted-loans file is large (over 2 million rows), so keep the raw CSV out of git and load it into a DuckDB file.

**Repository layout.**

```text
credit-risk-lifetime-ecl/
  README.md            question, findings, limits (written last, drafted early)
  requirements.txt
  data/raw/            not committed
  data/clean/          not committed; rebuilt by script
  sql/                 numbered scripts: 01_clean.sql, 02_vintage.sql ...
  src/                 python modules: features.py, pd_model.py, ecl.py
  notebooks/           one per stage, outputs cleared before commit
  tests/               pytest for key functions
  reports/             figures and the final memo
  Makefile             make all rebuilds everything
```

**Rules to set now.**

- Every stage ends with a commit and a short note of what you found, even if it was boring.
- Set a random seed everywhere, and split data by time, never at random.
- Keep a running `ASSUMPTIONS.md`. Every number you choose yourself (LGD, scenario paths, staging rule) goes there with a reason.

## Stage 1: Data, default definition, quality checks (weekend 1, morning)

**Goal:** one clean loan table where every row has a clear outcome and a known observation window.

**Steps**

1. Load the raw file into DuckDB and count rows, columns and missing values per column. Save the counts.
2. Keep fully observed loans only. Drop loans whose status is still open (Current, In Grace Period, Late) because their outcome is unknown. Write down how many you drop and why.
3. Define the target and write it in `ASSUMPTIONS.md`: default = Charged Off or Default; non-default = Fully Paid. Decide how to treat "Does not meet the credit policy" statuses and note the choice.
4. Remove columns that leak the outcome: payments received, recoveries, last payment date, anything recorded after origination. Keep a separate list of these, because Stage 2 needs them for return analysis but the PD model must never see them.
5. Create origination-time features only: loan amount, term, rate, grade, income, debt-to-income, employment length, home ownership, credit score range, open accounts, revolving utilisation, state.
6. Add the issue year and quarter, plus loan age in months at default or payoff, which Stage 4 needs.

**Data-quality checks to run and report** (a risk team expects this section)

- Duplicate loan ids.
- Impossible values: negative income, debt-to-income above a sane cap, dates out of order.
- Share of missing values per feature, and how you treat each (drop, impute, flag).
- Drift in the default rate by issue year, as a first plot.

**Checkpoint.** A clean table, a one-page data-quality note and a default-rate-by-year chart. If the default rate for the newest vintages looks suspiciously low, that is right-censoring, not good lending: those loans have not had time to default. Note it now, because it matters in every later stage.

**Pitfall.** Treating open loans as non-default quietly biases everything toward low default rates in recent years.

## Stage 2: Vintage analysis and realised return vs quoted rate (weekend 1, afternoon)

**Goal:** answer the headline question with data: did each grade earn its risk premium?

**Vintage analysis**

1. A vintage is all loans issued in one year (or quarter). For each vintage and grade, compute the cumulative default rate by loan age in months. Use only ages every loan in the vintage could have reached, to avoid censoring bias.
2. Plot default curves by vintage for each grade, one panel per grade. The question for each panel: do newer vintages default faster at the same age?
3. Note any vintage that looks different, and what was happening then (the 2015 to 2016 loosening of standards, the 2020 shock). Describe events as context, not as proven causes.

**Realised return versus quoted rate**

1. For each loan, build a simple realised return: (total payments received + recoveries - funded amount) divided by funded amount, then convert to an annualised figure using the loan's actual life. State the formula; it ignores reinvestment and servicing fees.
2. Group by grade and vintage. Compare the average quoted interest rate with the average annualised realised return.
3. Produce one table and one chart: grade on the x-axis, quoted rate and realised return side by side.
4. Report which grades earned their premium and which did not, and by how much. Add confidence intervals (bootstrap by loan) so small differences are not over-read.

**Checkpoint.** Two charts (vintage curves, quoted versus realised) and three sentences of findings. Write whatever the data says, including a boring answer.

**Pitfalls**

- Younger vintages have unfinished loans. Restrict the return analysis to loans that have reached final status, and say that this biases toward shorter, worse loans in recent years.
- Realised return is not IRR on a full cash-flow schedule. Name the simplification.
- Do not call a grade mispriced from one vintage; check it across several.

## Stage 3: Scorecard PD model and validation (weekend 2)

**Goal:** a 12-month-style default probability model that is interpretable, validated out of time and honestly benchmarked.

**Split by time**

- Train on earlier issue years, validate on a middle block, test on the latest years that have mostly reached final status. Fix the boundaries before you look at any results and write them in `ASSUMPTIONS.md`.
- Never shuffle rows across years.

**Build the scorecard**

1. Bin each numeric variable (start with quantile bins, then merge neighbours that do not differ). Compute weight of evidence (WoE) and information value (IV) per variable; drop variables with very low IV or implausibly high IV (a sign of leakage).
2. Check that WoE trends make business sense (higher debt-to-income should not lower risk). Fix or drop bins that break this.
3. Fit a logistic regression on the WoE-transformed features. Report coefficients and signs.
4. Convert to a score if you wish (points-to-double-odds scaling); it is optional but common in credit risk.

**Validate**

| Test | What it answers | Report |
| --- | --- | --- |
| AUC and Gini (2 x AUC - 1) | Does it rank risk? | Train, validation, test |
| KS statistic | Where is separation largest? | Test |
| Calibration plot | Do predicted PDs match observed default rates? | By decile, test |
| PSI (population stability index) | Has the borrower mix drifted? | Train vs each later year |
| Out-of-time drop | Does performance decay? | Gini by test year |

**Benchmark.** Fit a gradient boosting model with monotonic constraints on the same features. Compare Gini and calibration with the scorecard. Use SHAP to check that the model uses variables in a sensible way. Then write one honest paragraph: is the accuracy gain worth the loss in interpretability?

**Checkpoint (end of v1).** A validated scorecard, a comparison table, a calibration chart and a drift table. Commit, write the README draft and publish v1.

**Pitfalls**

- Leakage: any variable recorded after origination invalidates the model. Re-check column by column.
- Random splits inflate performance and hide drift.
- High Gini on a model with poor calibration is still unusable for provisioning. Report both.

## Stage 4: Lifetime PD with a survival model and macro variables (weeks 2 to 3)

**Goal:** estimate how default risk evolves with loan age and the economy, which is what lifetime ECL needs. This stage reuses the survival methods from your NIS2 thesis.

**Why survival, not a single PD.** A 12-month PD says nothing about a 36- or 60-month loan. A survival model estimates the hazard (the chance of defaulting in month t, given survival to t) and builds the lifetime PD from it. It also handles censoring: loans that were paid off early or are still open contribute information up to the point they leave.

**Steps**

1. Reshape to a loan-month table (discrete-time hazard): one row per loan per month at risk, with an event flag in the month of default. This is large, so build it in DuckDB and sample sensibly for modelling (for example by loan, keeping all months of each sampled loan).
2. Treat early payoff as censoring, not as a non-default outcome, and write that choice down.
3. Fit a discrete-time hazard model (logistic regression on loan-month rows) with loan age, borrower features and grade. As a check, fit a Cox model with `lifelines` on a loan-level table and compare the direction of effects.
4. Download monthly US unemployment and a short-term interest-rate series from FRED. Align them to the calendar month of each loan-month row, so macro conditions vary through a loan's life.
5. Add the macro variables and test whether they improve out-of-time fit. Compare with a model that has no macro variables. Report the likelihood-ratio test or the change in out-of-time log-loss.
6. Build the lifetime PD for a loan as 1 - the product of (1 - hazard) over its remaining months, under a stated macro path.

**Validate**

- Compare predicted and observed cumulative default curves by grade and vintage on held-out years.
- Report time-dependent performance: does the model rank risk equally well early and late in a loan's life?

**Checkpoint.** A hazard model with and without macro variables, a chart of predicted vs observed cumulative default by grade, and one paragraph: do macro conditions add information beyond borrower characteristics?

**Pitfalls**

- The macro series and defaults are both trending and autocorrelated. Do not read a correlation as a causal effect; say "associated with".
- With few distinct economic cycles in the sample, macro coefficients can be unstable. Test sensitivity to dropping a year.
- Lending Club is a US consumer platform, so US macro variables are the appropriate ones; do not mix in Turkish or Italian data.

## Stage 5: Scenario-weighted ECL (week 4)

**Goal:** turn the lifetime PD into an IFRS 9-style provision under several economic scenarios, and show how sensitive it is.

**The core formula**

```latex
ECL = \sum_{t=1}^{T} PD_t \times LGD \times EAD_t \times DF_t
```

PD\_t is the marginal default probability in month t from Stage 4, LGD is loss given default, EAD\_t is the balance outstanding at month t, and DF\_t discounts at the loan's effective rate. Stage 1 uses only the next 12 months; Stage 2 uses the full remaining life.

**Decisions to make and document**

| Item | Suggested approach | Where it goes |
| --- | --- | --- |
| LGD | Estimate from data if recoveries are available (1 - recoveries / defaulted balance, by grade); otherwise a stated flat value | ASSUMPTIONS.md |
| EAD | Amortisation schedule from loan terms; balance at default for defaulted loans | src/ecl.py |
| Staging | Stage 2 if grade is C or worse at origination, or if the Stage 4 PD has risen above a set multiple of its origination value. This rule is your own simplification | ASSUMPTIONS.md |
| Discount rate | The loan's interest rate | src/ecl.py |

**Scenarios**

1. Define three macro paths for unemployment and rates over the next 36 months: baseline, adverse, severe. Base them on historical episodes in the FRED data (for example, scale the observed 2008 to 2010 or 2020 path), and say they are your assumptions, not forecasts.
2. Run each path through the hazard model to get scenario PDs and scenario ECL.
3. Weight them (for example 60 / 30 / 10) and compute the weighted ECL. Report a table of ECL under each weight set, so the reader sees how much the provision moves when weights change.
4. Show ECL by grade and stage as a bar chart, scenario-weighted versus baseline.

**Checkpoint (end of v2).** A reproducible ECL calculation, a scenario table, a sensitivity table and a one-paragraph reading: how much extra provision does the adverse scenario require, and which grades drive it?

**Pitfalls**

- Averaging PDs across scenarios and then applying ECL differs from averaging ECL across scenarios. Because the relationship is non-linear, compute ECL per scenario, then weight.
- Do not present the output as a real provision. It illustrates the method on a US consumer portfolio with simplified staging and LGD.

## Stage 6: Profit-based approval cutoff (week 5)

**Goal:** replace "pick the threshold that maximises accuracy" with "pick the approval cutoff that maximises expected profit", and show how it moves in a downturn.

**Expected profit per loan**

```latex
E[profit] = (1 - PD) \times interest\ earned - PD \times LGD \times EAD - funding\ and\ servicing\ cost
```

Use the realised-return logic from Stage 2 to sanity-check the interest term. Funding and servicing cost is a stated assumption (a flat annual percentage), recorded in `ASSUMPTIONS.md`.

**Steps**

1. On the test years, score every loan with the Stage 3 or Stage 4 model.
2. Sort loans by predicted risk. For each possible cutoff, approve loans below it and compute total expected profit and the realised profit from actual outcomes. Plot both against the cutoff.
3. Pick the cutoff with the highest expected profit. Compare it with the realised-profit curve: how close is the model's choice to the best realised choice?
4. Compare the policy with two baselines: approve everyone, and approve by grade only (for example grades A to C).
5. Repeat under the adverse scenario from Stage 5 (higher PDs and, if you wish, higher LGD). Show how the best cutoff tightens.
6. Add a sensitivity table: best cutoff and profit for several LGD and funding-cost values.

**Checkpoint.** One chart (profit versus cutoff, expected and realised, baseline and downturn), one table comparing the three policies, and a short reading: how much profit does the model-based policy add, and how fragile is that result to the assumptions?

**Pitfalls**

- Selection bias: you only observe outcomes for loans that Lending Club approved. A policy evaluated on approved loans cannot tell you about the rejected ones. Say this explicitly; Stage 7 addresses it partly.
- Do not tune the cutoff on the test years and then report test performance. Choose it on the validation years, then apply it to test.
- Small profit differences are within noise. Bootstrap by loan to get intervals.

## Stage 7: Model risk, backtest and reject inference (weeks 5 to 6)

**Goal:** show exactly where the model can be trusted and where it cannot. This is the section that separates a tutorial from risk work.

**1. Model-risk checks**

- **Drift.** Compute PSI for each feature and for the score across years. List the features that drift most and what happens to performance when they do.
- **Where it breaks.** Train on the early years only and score each later year. Plot Gini and calibration error by year. Identify the year where calibration fails and the variables whose effect changes.
- **Champion versus challenger.** Compare the scorecard (champion) with the gradient boosting model (challenger) on stability, not only accuracy: which degrades faster out of time?
- **Stress backtest.** Take the adverse scenario you built from the 2008 to 2010 path or the 2020 path, and compare the scenario's predicted default uplift with what actually happened in the later vintages. Report the gap and call it what it is: a rough check, since the portfolio and period differ.
- **Calibration to the long-run level.** Check whether the model's average PD matches the through-the-cycle default rate, and show what changes if you rescale it.

**2. Reject inference (attempt, then judge)**

The model is trained only on accepted loans. Lending Club also published a file of rejected applications, with only a few fields (requested amount, score, debt-to-income, state, date, employment length; confirm from the file). Steps:

1. Check which variables exist in both files. If fewer than about three useful ones overlap, stop and write that the file is too thin; document this as a limitation.
2. Otherwise score the rejected applications with a model built only on the shared variables, assign each an estimated default probability, and add them (weighted) to the training data. This is a simple augmentation approach.
3. Compare cutoffs and profit with and without reject inference. Report how much the decision changes.
4. State clearly that reject inference rests on assumptions you cannot test here, and that the real default behaviour of rejected applicants is unobserved.

**3. Model-risk summary table**

End with one table: risk, evidence, severity, mitigation. For example: drift after the 2015 vintage, macro coefficient instability, LGD assumed rather than estimated, survivorship in approved-only data.

**Checkpoint (end of v3).** A drift report, a where-it-breaks chart, a backtest comparison, the reject-inference result (or the documented reason it was not done), and the model-risk table.

**Pitfalls**

- Do not hide a weak result. A clearly described failure is more credible than a perfect-looking model.
- If reject inference is thin, do not force it. Time is better spent on the write-up.

## Write-up, repository and how to use it

**The risk memo (3 pages).** Write it for a Chief Risk Officer, not for a classmate.

1. Page 1: the question, the answer in two or three sentences, and the main chart.
2. Page 2: method in plain language, validation results, and the ECL and sensitivity tables.
3. Page 3: assumptions, model-risk table and limitations.

Lead with the finding. State every number you chose yourself.

**README (top of the GitHub repo).** One-sentence question, one-sentence answer, three screenshots of charts, how to run it (`make all`), data sources with licences and download date, and a limitations section. Keep the repo free of raw data.

**Quality bar before you publish**

- A fresh clone runs end to end with one command.
- Every notebook has its outputs cleared or reproducible.
- Unit tests pass for default definition, hazard reshaping, ECL calculation and the profit function.
- No leaked columns, no random splits, no unexplained numbers.

**CV bullet, by version**

| Version finished | Bullet |
| --- | --- |
| v1 | Built and validated a PD scorecard on Lending Club loans (out-of-time Gini, PSI); vintage analysis of risk pricing by grade |
| v2 | Added discrete-time hazard model with US macro covariates and scenario-weighted IFRS 9-style lifetime ECL |
| v3 | Added profit-based approval cutoff, stress backtest and model-risk assessment |

Put the numbers in only once you have them, for example the out-of-time Gini or the ECL range across scenarios.

**LinkedIn and GitHub.** Pin the repository in your GitHub profile and the Featured section on LinkedIn. Add "Credit risk modelling", "Survival analysis" and "IFRS 9" as skills only after v2.

**Using it in conversations**

- With Prometeia (Nicolo Fasolato offered to answer questions): send the repository link and ask one specific question, such as how a junior team validates a PD model or what the Istanbul team works on. Do not ask for a job in the same message.
- In an interview, prepare a two-minute version: the question, the method, the one surprising result, the biggest limitation.
- Be ready to explain every choice, especially the default definition, the time split and the LGD assumption. Interviewers probe these.

## Schedule, checkpoints and common pitfalls

About 10 hours a week. If a week slips, cut scope from the end (Stage 7 first, then Stage 6), never from validation.

| Week | Work | Done when |
| --- | --- | --- |
| 1 (weekend) | Setup, Stage 1, Stage 2 | Clean table, vintage curves, quoted vs realised chart |
| 2 (weekend) | Stage 3, README draft | Validated scorecard; v1 published |
| 3 | Stage 4 data reshaping and hazard model | Hazard model with and without macro variables |
| 4 | Stage 5 | Scenario ECL and sensitivity table; v2 published |
| 5 | Stage 6 and start of Stage 7 | Profit-cutoff chart and policy comparison |
| 6 | Rest of Stage 7, memo, repo polish | v3 published, memo done |

**Rules for staying on track**

- Publish v1 before starting Stage 4. A finished smaller project beats an unfinished large one.
- If Stage 4 takes more than a week, simplify: fit the discrete-time hazard model only and skip the Cox comparison.
- Spend the last two days on the write-up, not on one more model.
- Timebox each stage; when it runs over, write what is missing as a limitation and move on.

**Mistakes that sink credit-risk projects**

| Mistake | Why it matters | What to do |
| --- | --- | --- |
| Using variables recorded after origination | Leakage inflates performance | Keep an explicit allowed-feature list |
| Random train and test split | Hides drift and overstates accuracy | Split by issue date |
| Treating open loans as non-default | Understates recent default rates | Drop them or treat as censored |
| Reporting only AUC | Ranking can be good while PDs are wrong | Always report calibration |
| Presenting assumed LGD or scenarios as findings | Misleads the reader | Label them assumptions everywhere |
| Claiming more than you built | Interviewers probe and it costs credibility | Use the claim table in the goal section |

**Questions to ask me along the way.** I can write starter code for any stage, review a notebook or the README, check a validation result for mistakes, or help you rehearse the two-minute interview version. Send me one stage at a time rather than the whole project.
