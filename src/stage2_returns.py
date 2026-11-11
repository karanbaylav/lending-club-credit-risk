"""Stage 2: vintage-curve chart and realised return vs quoted rate by grade.

Return sample: fully matured loans only (36-month loans issued 2007-2015, 60-month loans
issued 2007-2013), so every loan has had its full term and the sample is not tilted toward
early defaults and payoffs. Loans still open in that window (few) are excluded and counted."""
import numpy as np
import pandas as pd

from src.db import FIGS, SEED, TABLES, connect
from src.returns import loan_irr
from src.style import ACCENT, GRADE_COLORS, GREY, INK, WARN, plt, save

GRADES = list("ABCDEFG")
N_BOOT = 300


def vintage_chart():
    v = pd.read_csv(TABLES / "vintage_curves.csv")
    years = sorted(v.vintage.unique())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
    for ax, g in zip(axes.flat, GRADES):
        d = v[v.grade == g]
        top = 0
        for i, y in enumerate(years):
            s = d[d.vintage == y]
            if len(s) == 0 or s.loans.max() < 200:
                continue
            ax.plot(s.age, s.cum_default_pct, color=cmap(i / max(1, len(years) - 1)), lw=1.6)
            top = max(top, s.cum_default_pct.max())
        ax.set_title(f"Grade {g}")
        ax.set_ylim(0, top * 1.08 if top else 1)
        ax.set_xticks([0, 12, 24, 36])
    ax = axes.flat[-1]
    ax.axis("off")
    for i, y in enumerate(years):
        ax.plot([], [], color=cmap(i / max(1, len(years) - 1)), lw=2.5, label=str(y))
    ax.legend(title="Vintage (issue year)", frameon=False, ncol=2, loc="center")
    fig.supxlabel("Loan age (months)")
    fig.supylabel("Cumulative default rate (%)")
    fig.suptitle("Cumulative default by loan age, 36-month loans (vintages with 200+ loans)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, FIGS / "s2_vintage_curves.png")


def main():
    vintage_chart()
    con = connect(read_only=True)
    df = con.execute("""
        SELECT o.loan_id, f.grade, f.issue_year, o.term_months, o.funded_amnt, o.int_rate, o.installment,
               o.total_pymnt, o.recoveries, o.collection_recovery_fee, o.status_group, o.policy_fail,
               CASE WHEN o.status_group = 'default' THEN o.default_age_if_default - 1 ELSE o.last_pymnt_age END AS last_age
        FROM loan_outcomes o JOIN loan_features f USING (loan_id)
        WHERE (o.term_months = 36 AND f.issue_year <= 2015) OR (o.term_months = 60 AND f.issue_year <= 2013)
        ORDER BY o.loan_id
    """).df()
    con.close()
    excluded = int((df.status_group == "open").sum())
    df = df[df.status_group != "open"].copy()
    df["realised"] = loan_irr(df.funded_amnt, df.installment, df.total_pymnt, df.recoveries,
                              df.collection_recovery_fee, df.last_age)
    df["quoted"] = (1 + df.int_rate / 1200) ** 12 - 1           # quoted APR as an effective annual rate

    t = grade_table(df)
    t.round(2).to_csv(TABLES / "s2_return_by_grade.csv", index=False)
    # check: is the pattern a term effect? Same table on 36-month loans only (2007-2015)
    t36 = grade_table(df[df.term_months == 36])
    t36.round(2).to_csv(TABLES / "s2_return_by_grade_36m.csv", index=False)
    mix = (df.groupby("issue_year").agg(loans=("loan_id", "size"), pct_60_month=("term_months", lambda s: 100 * (s == 60).mean()))
             .reset_index())
    mix.round(1).to_csv(TABLES / "s2_term_mix_by_vintage.csv", index=False)

    by_v = vintage_table(df)
    by_v.round(2).to_csv(TABLES / "s2_return_by_grade_vintage.csv", index=False)
    vintage_table(df[df.term_months == 36]).round(2).to_csv(TABLES / "s2_return_by_grade_vintage_36m.csv", index=False)
    pd.DataFrame([{"matured_sample_loans": len(df), "open_loans_excluded": excluded,
                   "policy_fail_loans_in_sample": int(df.policy_fail.sum()),
                   "loans_36_month": int((df.term_months == 36).sum())}]).to_csv(TABLES / "s2_return_sample.csv", index=False)
    chart(t)
    print(t.round(2).to_string(index=False))
    print(t36.round(2).to_string(index=False))


def grade_table(df):
    rng = np.random.default_rng(SEED)
    rows = []
    for g, d in df.groupby("grade"):
        w, x = d.funded_amnt.to_numpy(), d.realised.to_numpy()
        point = np.average(x, weights=w)
        boots = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(d), len(d))
            boots.append(np.average(x[i], weights=w[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(grade=g, loans=len(d), default_rate_pct=100 * (d.status_group == "default").mean(),
                         quoted_pct=100 * np.average(d.quoted, weights=w), realised_pct=100 * point,
                         realised_lo_pct=100 * lo, realised_hi_pct=100 * hi))
    t = pd.DataFrame(rows)
    t["loss_of_return_pts"] = t.quoted_pct - t.realised_pct
    t["premium_over_A_pts"] = t.realised_pct - t.loc[t.grade == "A", "realised_pct"].iloc[0]
    return t


def vintage_table(df):
    by_v = (df.assign(wr=df.realised * df.funded_amnt, wq=df.quoted * df.funded_amnt)
              .groupby(["grade", "issue_year"])
              .agg(loans=("loan_id", "size"), wr=("wr", "sum"), wq=("wq", "sum"), w=("funded_amnt", "sum"))
              .reset_index())
    by_v["quoted_pct"] = 100 * by_v.wq / by_v.w
    by_v["realised_pct"] = 100 * by_v.wr / by_v.w
    return by_v[by_v.loans >= 200][["grade", "issue_year", "loans", "quoted_pct", "realised_pct"]]


def chart(t):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.quoted_pct, 0.38, color=GREY, label="Quoted interest rate (annual, effective)")
    ax.bar(x + 0.2, t.realised_pct, 0.38, color=ACCENT, label="Realised return (annualised, 95% CI)")
    ax.errorbar(x + 0.2, t.realised_pct, yerr=[t.realised_pct - t.realised_lo_pct, t.realised_hi_pct - t.realised_pct],
                fmt="none", ecolor=INK, lw=1.2, capsize=3)
    for xi, r, hi in zip(x, t.realised_pct, t.realised_hi_pct):
        ax.text(xi + 0.2, max(hi, 0) + 0.5, f"{r:.1f}%", ha="center", fontsize=9, color=ACCENT if r >= 0 else WARN)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [f"Grade {g}" for g in t.grade])
    ax.set_ylabel("% per year")
    ax.set_title("Quoted rate vs realised return by grade (fully matured loans)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, FIGS / "s2_quoted_vs_realised.png")


if __name__ == "__main__":
    main()
