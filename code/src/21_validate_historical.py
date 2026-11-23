"""Validation, 1994-2015: does the index move with established measures of (policy) uncertainty?

    ECON_RUN=hist .venv/bin/python src/21_validate_historical.py [--weekly PATH]

Same approach as section 5 of the pilot report: correlations in levels and in changes, for each
model's index and the no-AI word count. Changes are the stricter test, because two series that both
trend over 21 years correlate in levels even if they are unrelated. Level correlations of slow-moving
series are also inflated by autocorrelation, so the 5% cut-off shown for them is generous.

Comparison series (all public):
  weekly   VIX (FRED VIXCLS, weekly mean); 2-year Treasury yield volatility (FRED DGS2: standard
           deviation of daily changes within the week, bp) as a market measure of rate uncertainty
  monthly  Husted-Rogers-Sun monetary policy uncertainty (newspaper-based, specifically about the Fed);
           Baker-Bloom-Davis monetary-policy EPU; CPI inflation (y/y); unemployment rate
Monthly values of the index are averages of its weekly values.
Output: output/1994_2015/validation.txt
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT, RAW, RUN

assert RUN == "hist"
F = RAW / "hist_fred"


def fred(s):
    d = pd.read_csv(F / f"fred_{s}.csv")
    d.columns = ["date", s]
    d["date"] = pd.to_datetime(d["date"])
    d[s] = pd.to_numeric(d[s], errors="coerce")
    return d.set_index("date")[s].dropna()


def crit(n):
    t = stats.t.ppf(0.975, n - 2)
    return t / np.sqrt(n - 2 + t ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", default=str(OUTPUT / "fed_uncertainty_weekly_1994_2015.csv"))
    args = ap.parse_args()
    w = pd.read_csv(args.weekly, index_col=0, parse_dates=True)
    models = [c.removesuffix("_policy") for c in w.columns if c.endswith("_policy")]
    idx = {f"{m}": w[f"{m}_policy"] for m in models} | {f"{m} (min+stmt)": w[f"{m}_policy_ms"] for m in models} \
        | {"word count": w["dictionary"]}

    # weekly comparison series, on the same Fridays
    vix = fred("VIXCLS").resample("W-FRI").mean()
    dgs2 = fred("DGS2")
    vol2 = (dgs2.diff() * 100).resample("W-FRI").std()
    weekly = pd.DataFrame({"VIX": vix, "2-yr yield volatility": vol2}).reindex(w.index)

    # monthly comparison series
    hrs = pd.read_excel(RAW / "hrs_mpu_monthly.xlsx").dropna()
    hrs = hrs[hrs["month"].astype(str).str.match(r"^\d{4}m\d{1,2}$")]
    hrs.index = pd.to_datetime(hrs["month"].str.replace("m", "-") + "-01")
    epu = pd.read_excel(RAW / "categorical_epu.xlsx").dropna(subset=["Year", "Month"])
    epu = epu[pd.to_numeric(epu["Year"], errors="coerce").notna()]
    epu.index = pd.to_datetime(epu["Year"].astype(int).astype(str) + "-" + epu["Month"].astype(int).astype(str) + "-01")
    cpi = fred("CPIAUCSL")
    monthly = pd.DataFrame({
        "Husted-Rogers-Sun Fed policy uncertainty": pd.to_numeric(hrs["US MPU"], errors="coerce"),
        "Baker-Bloom-Davis monetary policy EPU": pd.to_numeric(epu["2. Monetary policy"], errors="coerce"),
        "CPI inflation (y/y)": cpi.pct_change(12) * 100,
        "Unemployment rate": fred("UNRATE"),
    })
    mi = pd.DataFrame(idx).resample("MS").mean()
    monthly = monthly.reindex(mi.index)

    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    def table(left, right, label):
        rows = []
        for col in right.columns:
            ok = left.notna().all(axis=1) & right[col].notna()
            n = int(ok.sum())
            dk = left.diff().notna().all(axis=1) & right[col].diff().notna()
            row = {"series": col, "n": n}
            for name in left.columns:
                row[f"{name} level"] = left.loc[ok, name].corr(right.loc[ok, col])
                row[f"{name} change"] = left[name].diff()[dk].corr(right[col].diff()[dk])
            rows.append(row)
        t = pd.DataFrame(rows).set_index("series")
        n = int(t["n"].min())
        say(f"--- {label} (n about {n}; 5% cut-off |r| > {crit(n):.2f}, for changes n-1)")
        for name in left.columns:
            say(f"  {name:22s} " + " | ".join(f"{col[:34]:34s} level {t.loc[col, f'{name} level']:+.2f}  change "
                                             f"{t.loc[col, f'{name} change']:+.2f}" for col in right.columns))
        return t

    say(f"Index file: {args.weekly}")
    say(f"Sample: {w.index[0].date()} to {w.index[-1].date()} ({len(w)} weeks)\n")
    tw = table(pd.DataFrame(idx).reindex(w.index), weekly, "Weekly")
    say()
    tm = table(mi, monthly, "Monthly")
    say()
    # headline: correlations with the most direct benchmark, with confidence intervals
    say("--- Main index vs Husted-Rogers-Sun (the benchmark closest to what the index measures)")
    for name in idx:
        ok = mi[name].notna() & monthly.iloc[:, 0].notna()
        r = mi.loc[ok, name].corr(monthly.loc[ok].iloc[:, 0])
        n = int(ok.sum())
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        r12 = mi.loc[ok, name].rolling(12).mean().corr(monthly.loc[ok].iloc[:, 0].rolling(12).mean())
        say(f"  {name:22s} monthly level r = {r:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}, n={n}); 12-month averages r = {r12:+.2f}")
    (OUTPUT / "validation.txt").write_text("\n".join(lines) + "\n")
    pd.concat({"weekly": tw, "monthly": tm}).to_csv(OUTPUT / "validation_correlations.csv")
    print(f"\nSaved {OUTPUT / 'validation.txt'}")


if __name__ == "__main__":
    main()
