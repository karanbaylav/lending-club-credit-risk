"""Model data: origination-time features only, the 12-month target, and the time split.

The allowed-feature list below is the only way columns reach a model. Lending Club's own
pricing (interest rate, grade, sub-grade, instalment) is kept out of the borrower scorecard
so the scorecard can be compared with Lending Club's grades; `GRADE_FEATURES` adds it back
for a separate comparison."""
import numpy as np
import pandas as pd

from src.db import connect

NUMERIC = ["loan_amnt", "term_months", "annual_inc", "dti", "emp_length_yrs", "fico", "inq_last_6mths",
           "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "pub_rec_bankruptcies",
           "credit_history_months", "delinq_2yrs", "loan_to_income"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "application_type", "addr_state"]
GRADE_FEATURES = ["sub_grade"]
# mort_acc (number of mortgage accounts) was not recorded before 2012: 100% missing in 2007-2011,
# so its "missing" bin vanishes after training and its PSI explodes. Dropped from the models;
# stage 3 reports the scorecard with it as a sensitivity check.
DROPPED = {"mort_acc": "not recorded before 2012 (structural break, not drift)"}

# expected direction of the default rate as the variable rises (+1 up, -1 down, 0 no prior)
DIRECTION = {"loan_amnt": 0, "term_months": 1, "annual_inc": -1, "dti": 1, "emp_length_yrs": -1, "fico": -1,
             "inq_last_6mths": 1, "open_acc": 0, "pub_rec": 1, "revol_bal": 0, "revol_util": 1, "total_acc": -1,
             "mort_acc": -1, "pub_rec_bankruptcies": 1, "credit_history_months": -1, "delinq_2yrs": 1,
             "loan_to_income": 1}

SPLIT = {"train": (2007, 2014), "validation": (2015, 2015), "test": (2016, 2100)}


def load_model_frame(columns_extra=()):
    con = connect(read_only=True)
    extra = "".join(f", f.{c}" for c in columns_extra)
    df = con.execute(f"""
        SELECT f.loan_id, f.issue_month, f.issue_year, e.bad_12m::INT AS bad,
               f.loan_amnt, f.term_months, f.annual_inc, f.dti, f.emp_length_yrs, f.fico, f.inq_last_6mths,
               f.open_acc, f.pub_rec, f.revol_bal, f.revol_util, f.total_acc, f.mort_acc, f.pub_rec_bankruptcies,
               f.credit_history_months, f.delinq_2yrs, f.home_ownership, f.verification_status, f.purpose,
               f.application_type, f.addr_state, f.sub_grade, f.grade, f.int_rate {extra}
        FROM loan_features f JOIN loan_events e USING (loan_id)
        WHERE e.obs_12m
        ORDER BY f.loan_id
    """).df()
    con.close()
    return clean(df)


def clean(df):
    df = df.copy()
    df.loc[df.annual_inc <= 0, "annual_inc"] = np.nan
    df.loc[(df.dti < 0) | (df.dti > 100), "dti"] = np.nan
    df.loc[(df.fico < 300) | (df.fico > 850), "fico"] = np.nan
    df["revol_util"] = df.revol_util.clip(upper=150)
    df.loc[df.credit_history_months < 0, "credit_history_months"] = np.nan
    df["loan_to_income"] = df.loan_amnt / df.annual_inc
    for c in CATEGORICAL:
        df[c] = df[c].fillna("MISSING").astype(str)
    df["split"] = np.select([df.issue_year.between(*SPLIT["train"]), df.issue_year.between(*SPLIT["validation"])],
                            ["train", "validation"], "test")
    return df
