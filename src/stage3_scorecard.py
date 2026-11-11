"""Stage 3: WoE scorecard for 12-month default, out-of-time validation, monotonic GBM benchmark."""
import json
import os
import pathlib

import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit, logit

from src.db import FIGS, ROOT, SEED, TABLES
from src.features import CATEGORICAL, DIRECTION, DROPPED, NUMERIC, load_model_frame
from src.style import ACCENT, GREY, INK, WARN, plt, save
from src.validation import calibration_table, gini, ks, psi, psi_categorical
from src.woe import CategoricalBinner, NumericBinner

IV_MIN, IV_SUSPECT = 0.02, 0.5
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50      # 600 points = 50:1 good:bad, odds double every 20 points


def fit_binners(train):
    binners, rows = {}, []
    for c in NUMERIC:
        b = NumericBinner(direction=DIRECTION[c]).fit(train[c], train.bad)
        binners[c] = b
    for c in CATEGORICAL:
        binners[c] = CategoricalBinner().fit(train[c], train.bad)
    for c, b in binners.items():
        rows.append(dict(feature=c, iv=b.iv, bins=len(b.table), kind="numeric" if c in NUMERIC else "categorical",
                         direction_prior=DIRECTION.get(c, 0)))
    iv = pd.DataFrame(rows).sort_values("iv", ascending=False)
    iv["status"] = np.where(iv.iv < IV_MIN, "dropped: IV below 0.02",
                    np.where(iv.iv > IV_SUSPECT, "check: IV above 0.5 (possible leakage)", "kept"))
    return binners, iv


def woe_frame(df, binners, cols):
    return pd.DataFrame({c: binners[c].transform(df[c]) for c in cols}, index=df.index)


def fit_logit(X, y):
    """Fit, then drop the weakest wrong-signed variable until every coefficient is positive."""
    cols, dropped = list(X.columns), []
    while True:
        m = sm.Logit(y, sm.add_constant(X[cols])).fit(disp=0, maxiter=200)
        neg = m.params.drop("const")
        neg = neg[neg <= 0]
        if neg.empty:
            return m, cols, dropped
        worst = neg.abs().idxmin()
        cols.remove(worst)
        dropped.append(worst)


def to_score(p):
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return offset + factor * np.log((1 - p) / p)


def subgrade_rank(s):
    return s.map(lambda v: (ord(v[0]) - 65) * 5 + int(v[1]) if isinstance(v, str) and len(v) == 2 else np.nan)


def main():
    df = load_model_frame()
    tr, va, te = (df[df.split == s] for s in ("train", "validation", "test"))
    counts = df.groupby("split").agg(loans=("bad", "size"), bad_rate=("bad", "mean"),
                                     first_issue=("issue_month", "min"), last_issue=("issue_month", "max"))
    counts.to_csv(TABLES / "s3_split_counts.csv")

    binners, iv = fit_binners(tr)
    iv.round(4).to_csv(TABLES / "s3_information_value.csv", index=False)
    pd.concat([b.table.assign(feature=c) for c, b in binners.items()]).round(4).to_csv(TABLES / "s3_woe_bins.csv", index=False)
    keep = iv.loc[iv.status != "dropped: IV below 0.02", "feature"].tolist()

    Xtr = woe_frame(tr, binners, keep)
    model, cols, wrong_sign = fit_logit(Xtr, tr.bad)
    coef = pd.DataFrame({"feature": model.params.index, "coef": model.params.values, "se": model.bse.values,
                         "p_value": model.pvalues.values})
    coef.round(4).to_csv(TABLES / "s3_scorecard_coefficients.csv", index=False)

    def predict(d):
        return model.predict(sm.add_constant(woe_frame(d, binners, cols), has_constant="add"))

    p = {"train": predict(tr), "validation": predict(va), "test": predict(te)}

    # Lending Club's own sub-grade as a ranking benchmark, and a scorecard + sub-grade model
    sg_tr = subgrade_rank(tr.sub_grade)
    sg = {k: subgrade_rank(d.sub_grade) for k, d in (("train", tr), ("validation", va), ("test", te))}
    # scorecard + Lending Club sub-grade: does the borrower scorecard add anything to the grade?
    sg_binner = CategoricalBinner(min_share=0.0005).fit(tr.sub_grade, tr.bad)
    def plus_grade_X(d):
        X = woe_frame(d, binners, cols)
        X["sub_grade"] = sg_binner.transform(d.sub_grade)
        return sm.add_constant(X, has_constant="add")
    combo = sm.Logit(tr.bad, plus_grade_X(tr)).fit(disp=0, maxiter=200)
    pc = {k: combo.predict(plus_grade_X(d)) for k, d in (("train", tr), ("validation", va), ("test", te))}

    # Gradient boosting benchmark, monotone where the direction is known, early-stopped on validation
    feats = NUMERIC + CATEGORICAL
    def lgb_frame(d):
        X = d[feats].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=sorted(tr[c].unique()))
        return X
    mono = [DIRECTION[c] for c in NUMERIC] + [0] * len(CATEGORICAL)
    # small regularisation grid; every setting early-stops on 2015, and the pick uses 2015 only
    grid = [
        dict(name="default-ish", num_leaves=31, min_child_samples=500, reg_lambda=0.0, colsample_bytree=0.8),
        dict(name="regularised", num_leaves=15, min_child_samples=2000, reg_lambda=10.0, colsample_bytree=0.6),
        dict(name="strongly regularised", num_leaves=7, min_child_samples=5000, reg_lambda=50.0, colsample_bytree=0.5),
    ]
    Xtr_l, Xva_l = lgb_frame(tr), lgb_frame(va)
    tuning, fitted = [], {}
    cache = pathlib.Path(os.environ.get("LC_CACHE", ROOT / "data" / "clean" / "gbm_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    for g in grid:
        m_ = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                monotone_constraints=mono, monotone_constraints_method="advanced",
                                random_state=SEED, verbose=-1, **{k: v for k, v in g.items() if k != "name"})
        f = cache / (g["name"].replace(" ", "_") + ".pkl")
        if f.exists():                      # fitted earlier with the same data and seed
            m_ = joblib.load(f)
        else:
            m_.fit(Xtr_l, tr.bad, eval_set=[(Xva_l, va.bad)], eval_metric="auc",
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            joblib.dump(m_, f)
        fitted[g["name"]] = m_
        tuning.append(dict(setting=g["name"], trees=m_.best_iteration_,
                           gini_train=gini(tr.bad, m_.predict_proba(Xtr_l)[:, 1]),
                           gini_validation=gini(va.bad, m_.predict_proba(Xva_l)[:, 1]), **g))
    tuning = pd.DataFrame(tuning)
    best = tuning.sort_values("gini_validation", ascending=False).setting.iloc[0]
    tuning["chosen_on_validation"] = tuning.setting == best
    gbm = fitted[best]
    q = {k: gbm.predict_proba(lgb_frame(d))[:, 1] for k, d in (("train", tr), ("validation", va), ("test", te))}
    tuning["gini_test_chosen_only"] = np.where(tuning.chosen_on_validation, gini(te.bad, q["test"]), np.nan)
    tuning.drop(columns="name").round(4).to_csv(TABLES / "s3_gbm_tuning.csv", index=False)

    rows = []
    for k, d in (("train", tr), ("validation", va), ("test", te)):
        rows.append(dict(sample=k, loans=len(d), bad_rate=d.bad.mean(),
                         gini_scorecard=gini(d.bad, p[k]), gini_gbm=gini(d.bad, q[k]),
                         gini_lc_subgrade=gini(d.bad[sg[k].notna()], sg[k].dropna()),
                         gini_scorecard_plus_subgrade=gini(d.bad, pc[k]),
                         ks_scorecard=ks(d.bad, p[k]), ks_gbm=ks(d.bad, q[k]),
                         mean_pd_scorecard=float(np.mean(p[k])), mean_pd_gbm=float(np.mean(q[k]))))
    perf = pd.DataFrame(rows)
    perf.round(4).to_csv(TABLES / "s3_performance.csv", index=False)

    by_year = []
    for y, d in df[df.split != "train"].groupby("issue_year"):
        k = "validation" if y == 2015 else "test"
        idx = d.index
        src = va if k == "validation" else te
        pp = pd.Series(p[k], index=src.index).loc[idx]
        qq = pd.Series(q[k], index=src.index).loc[idx]
        sgy = subgrade_rank(d.sub_grade)
        by_year.append(dict(issue_year=y, loans=len(d), bad_rate=d.bad.mean(), gini_scorecard=gini(d.bad, pp),
                            gini_lc_subgrade=gini(d.bad[sgy.notna()], sgy.dropna()),
                            gini_gbm=gini(d.bad, qq), mean_pd_scorecard=pp.mean(), mean_pd_gbm=qq.mean(),
                            psi_score_vs_train=psi(to_score(p["train"]), to_score(pp))))
    by_year = pd.DataFrame(by_year)
    by_year.round(4).to_csv(TABLES / "s3_by_year.csv", index=False)

    # feature drift: PSI of each WoE feature, train vs the test years
    Xte = woe_frame(te, binners, cols)
    drift = pd.DataFrame([dict(feature=c, psi_train_vs_test=psi_categorical(Xtr[c], Xte[c])) for c in cols]
                         ).sort_values("psi_train_vs_test", ascending=False)
    drift.round(4).to_csv(TABLES / "s3_feature_psi.csv", index=False)

    cal = calibration_table(te.bad.to_numpy(), p["test"].to_numpy()).assign(model="scorecard")
    cal_g = calibration_table(te.bad.to_numpy(), q["test"]).assign(model="gbm")
    pd.concat([cal, cal_g]).round(5).to_csv(TABLES / "s3_calibration_test.csv", index=False)

    # SHAP on a test sample: does the GBM use each variable in a sensible direction?
    import shap
    samp = te.sample(min(20000, len(te)), random_state=SEED)
    sv = shap.TreeExplainer(gbm).shap_values(lgb_frame(samp))
    sv = sv[1] if isinstance(sv, list) else sv
    shap_tab = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)})
    shap_tab["corr_value_vs_shap"] = [np.corrcoef(samp[c].fillna(samp[c].median()), sv[:, i])[0, 1] if c in NUMERIC else np.nan
                                      for i, c in enumerate(feats)]
    shap_tab.sort_values("mean_abs_shap", ascending=False).round(4).to_csv(TABLES / "s3_gbm_shap.csv", index=False)

    # calibration drift: mean predicted vs observed 12-month default rate for every issue year
    all_p = predict(df)
    all_q = gbm.predict_proba(lgb_frame(df))[:, 1]
    cy = (df.assign(p_sc=all_p, p_gbm=all_q).groupby(["issue_year", "split"])
            .agg(loans=("bad", "size"), observed=("bad", "mean"), predicted_scorecard=("p_sc", "mean"),
                 predicted_gbm=("p_gbm", "mean")).reset_index())
    cy["observed_over_predicted"] = cy.observed / cy.predicted_scorecard
    # intercept shift fitted on the 2015 validation year only, then applied unchanged to the test years
    lv = logit(np.clip(p["validation"].to_numpy(), 1e-9, 1 - 1e-9))
    shift = brentq(lambda s: expit(lv + s).mean() - va.bad.mean(), -3, 3)
    lt = logit(np.clip(p["test"].to_numpy(), 1e-9, 1 - 1e-9))
    p_shift = expit(lt + shift)
    cy["predicted_scorecard_shifted"] = np.where(cy.split == "test", np.nan, np.nan)
    for y in cy.issue_year[cy.split == "test"]:
        m_ = (te.issue_year == y).to_numpy()
        cy.loc[cy.issue_year == y, "predicted_scorecard_shifted"] = p_shift[m_].mean()
    cy.round(5).to_csv(TABLES / "s3_calibration_by_year.csv", index=False)
    calib = pd.DataFrame([dict(item="intercept shift fitted on 2015 (log-odds)", value=shift),
                          dict(item="test observed 12m default rate", value=te.bad.mean()),
                          dict(item="test mean PD, scorecard as trained", value=p["test"].mean()),
                          dict(item="test mean PD, scorecard after 2015 shift", value=p_shift.mean()),
                          dict(item="share of the test gap closed by the shift",
                               value=(p_shift.mean() - p["test"].mean()) / (te.bad.mean() - p["test"].mean()))])
    calib.round(5).to_csv(TABLES / "s3_intercept_shift.csv", index=False)
    calibration_table(te.bad.to_numpy(), p_shift).round(5).to_csv(TABLES / "s3_calibration_test_shifted.csv", index=False)

    # sensitivity: the scorecard with mortgage accounts (dropped from the main model)
    b_m = NumericBinner(direction=DIRECTION["mort_acc"]).fit(tr.mort_acc, tr.bad)
    def with_m(d):
        X = woe_frame(d, binners, cols)
        X["mort_acc"] = b_m.transform(d.mort_acc)
        return sm.add_constant(X, has_constant="add")
    m_m = sm.Logit(tr.bad, with_m(tr)).fit(disp=0, maxiter=200)
    sens = pd.DataFrame([dict(model="scorecard without mort_acc (main)", iv_mort_acc=np.nan,
                              gini_validation=gini(va.bad, p["validation"]), gini_test=gini(te.bad, p["test"])),
                         dict(model="scorecard with mort_acc", iv_mort_acc=b_m.iv,
                              gini_validation=gini(va.bad, m_m.predict(with_m(va))), gini_test=gini(te.bad, m_m.predict(with_m(te))),
                              psi_mort_acc_woe_train_vs_test=psi_categorical(b_m.transform(tr.mort_acc), b_m.transform(te.mort_acc)),
                              share_missing_train=tr.mort_acc.isna().mean(), share_missing_test=te.mort_acc.isna().mean())])
    sens.round(4).to_csv(TABLES / "s3_sensitivity_mort_acc.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(cy.issue_year, 100 * cy.observed, marker="o", color=INK, label="Observed")
    ax.plot(cy.issue_year, 100 * cy.predicted_scorecard, marker="o", color=ACCENT, label="Scorecard, as trained")
    ts = cy[cy.split == "test"]
    ax.plot(ts.issue_year, 100 * ts.predicted_scorecard_shifted, marker="D", color=WARN, ls="--",
            label="Scorecard after an intercept shift fitted on 2015")
    for s, c in (("validation", "#EEF3FA"), ("test", "#FBEFE8")):
        yrs = cy.issue_year[cy.split == s]
        ax.axvspan(yrs.min() - 0.5, yrs.max() + 0.5, color=c, lw=0, zorder=0)
    ax.set_xticks(cy.issue_year)
    ax.set_ylim(0, 100 * cy.observed.max() * 1.25)
    for s, x0 in (("validation", 2015), ("test", 2016.5)):
        ax.text(x0, 100 * cy.observed.max() * 1.2, s, ha="center", fontsize=9, color=INK)
    ax.set_ylabel("12-month default rate (%)")
    ax.set_title("Predicted vs observed 12-month default rate by issue year")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    save(fig, FIGS / "s3_calibration_by_year.png")

    meta = dict(features_kept_after_iv=keep, wrong_sign_dropped=wrong_sign, final_features=cols,
                gbm_setting=best, gbm_trees=int(gbm.best_iteration_ or gbm.n_estimators), dropped_features=DROPPED, seed=SEED)
    (TABLES / "s3_model_meta.json").write_text(json.dumps(meta, indent=2))

    # charts
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    m = max(cal.predicted_pd.max(), cal.observed_rate.max(), cal_g.observed_rate.max()) * 1.08
    ax.plot([0, m], [0, m], color=GREY, lw=1, ls="--")
    ax.plot(cal.predicted_pd, cal.observed_rate, marker="o", color=ACCENT, label="Scorecard")
    ax.plot(cal_g.predicted_pd, cal_g.observed_rate, marker="s", color=WARN, label="Gradient boosting")
    ax.set_xlabel("Predicted 12-month PD (decile mean)")
    ax.set_ylabel("Observed 12-month default rate")
    ax.set_title("Calibration on the test years")
    ax.legend(frameon=False)
    save(fig, FIGS / "s3_calibration_test.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(by_year.issue_year, by_year.gini_scorecard, marker="o", color=ACCENT, label="Scorecard")
    ax.plot(by_year.issue_year, by_year.gini_gbm, marker="s", color=WARN, label="Gradient boosting")
    ax.plot(by_year.issue_year, by_year.gini_lc_subgrade, marker="^", color=GREY, label="Lending Club sub-grade")
    ax.axhline(perf.loc[perf["sample"] == "train", "gini_scorecard"].iloc[0], color=ACCENT, ls=":", lw=1)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(by_year.issue_year)
    ax.set_ylabel("Gini")
    ax.set_title("Out-of-time Gini by issue year (dotted: scorecard on train)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, FIGS / "s3_gini_by_year.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ivp = iv.sort_values("iv")
    ax.barh(ivp.feature, ivp.iv, color=[ACCENT if s == "kept" else GREY for s in ivp.status])
    ax.axvline(IV_MIN, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("Information value (train)")
    ax.set_title("Information value per feature (grey = dropped)")
    save(fig, FIGS / "s3_information_value.png")

    print(perf.round(3).to_string(index=False))
    print(by_year.round(3).to_string(index=False))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
