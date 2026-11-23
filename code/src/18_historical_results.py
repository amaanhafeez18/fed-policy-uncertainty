"""Historical run, results: weekly Fed policy-uncertainty index on the weekly Friday dates (1993-12-31 to 2015-05-29).

    ECON_RUN=hist .venv/bin/python src/18_historical_results.py            # needs every model >= 95% scored
    ECON_RUN=hist .venv/bin/python src/18_historical_results.py --partial  # test on whatever is scored so far

Same index method as the pilot (05_build_index.py):
  1. paragraph -> document: average policy_uncertainty over the paragraphs the model marks as about the
     future policy path (each document counts once, however long it is);
  2. document -> week (weeks end Friday): average over documents released that week;
  3. a week with no new policy text keeps the previous week's value (values are carried forward from
     documents released before the sample, so the first week has a value).
Every paragraph is scored and kept; non-policy paragraphs simply have no policy uncertainty to average.

Built for each model (Luna, Gemini, and Haiku once scored):
  policy        the main index (all four document types)
  policy_ms     check version: minutes and statements only, the two types that exist for the whole
                sample (speeches start 1996, press conferences April 2011)
  economic      uncertainty about the economy, averaged over ALL paragraphs (nothing left out)
  policy_share  share of that week's paragraphs that are about future policy
plus the no-AI word-count benchmark (dictionary).

Outputs (output/1994_2015/):
  fed_uncertainty_weekly_1994_2015.csv / .xlsx   one row per week (the .xlsx adds notes and coverage)
  fig_index_by_model.png                         one panel per model
  fig_models_compared.png                        13-week averages of every model, plus the check version
  checks.txt                                     score checks, model agreement, known-history table
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT, PROCESSED, RAW, RUN

assert RUN == "hist", "run with ECON_RUN=hist"

MODELS = {  # file tag -> short name, label, colour (validated colourblind-safe order from the pilot)
    "gpt-6-luna": ("luna", "GPT-6 Luna (OpenAI)", "#2a78d6"),
    "gemini-3.1-flash-lite": ("gemini", "Gemini 3.1 Flash-Lite (Google)", "#eb6834"),
    "claude-haiku-5-5": ("haiku", "Claude Haiku 5.5 (Anthropic)", "#1baf7a"),
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
RECESSIONS = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30")]  # NBER
EVENTS = {"1994-02-04": "First hike\nFeb 94", "1998-09-29": "LTCM\ncuts", "2001-01-03": "Easing\nbegins",
          "2001-09-17": "9/11", "2003-08-12": "\"Considerable\nperiod\"", "2004-06-30": "\"Measured\npace\"",
          "2007-08-17": "Credit\ncrunch", "2008-09-16": "Lehman", "2008-12-16": "Zero\nbound",
          "2011-08-09": "\"Mid-2013\"", "2013-05-22": "Taper\ntantrum", "2014-12-17": "\"Patient\""}
# Periods where Fed history gives a clear expectation, used only to CHECK the index afterwards
# (nothing in the method is tuned to them).
HISTORY = [
    ("1994-02-04", "1995-02-01", "1994 tightening cycle", "uncertain: surprise hikes, no guidance"),
    ("1998-08-17", "1998-11-17", "Russia/LTCM crisis, three cuts", "higher"),
    ("2001-01-03", "2001-12-11", "2001 easing, 11 cuts", "higher"),
    ("2003-08-12", "2004-06-29", "'considerable period' guidance", "lower: explicit guidance"),
    ("2004-06-30", "2006-06-29", "'measured pace' hikes, 17 x 25bp", "lowest: path pre-announced"),
    ("2007-08-10", "2008-12-15", "financial crisis, cuts to zero", "higher"),
    ("2008-12-16", "2011-08-08", "zero bound, 'extended period'", "moderate"),
    ("2011-08-09", "2012-12-11", "calendar guidance ('mid-2013', 'late 2014')", "lower: dated commitment"),
    ("2012-12-12", "2013-12-17", "thresholds; taper debate", "higher in mid-2013"),
    ("2013-12-18", "2015-05-29", "taper, then 'patient'", "moderate"),
]


def load(partial: bool):
    p = pd.read_csv(PROCESSED / "passages.csv", parse_dates=["date"])
    p = p.merge(pd.read_csv(PROCESSED / "scores_dictionary.csv"), on="passage_id")
    models, log = {}, []
    for tag, (short, label, color) in MODELS.items():
        path = PROCESSED / f"scores_llm_{tag}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        ok = pd.DataFrame([r for r in rows if "error" not in r]).drop_duplicates("passage_id", keep="last")
        errs = pd.Series([r["error"] for r in rows if "error" in r], dtype=str)
        cov = p["passage_id"].isin(ok["passage_id"]).mean()
        log.append(dict(model=tag, scored=len(ok), coverage=cov, error_rows=len(errs),
                        errors=errs.value_counts().to_dict(), in_tok=int(ok.input_tokens.sum()),
                        out_tok=int(ok.output_tokens.sum())))
        if cov < 0.95 and not partial:
            print(f"Skipping {tag}: only {cov:.1%} scored")
            continue
        s = ok[["passage_id", "policy_relevant", "policy_uncertainty", "economic_uncertainty", "stance", "reason"]]
        p = p.merge(s.add_prefix(f"{short}_").rename(columns={f"{short}_passage_id": "passage_id"}),
                    on="passage_id", how="left")
        models[short] = (label, color)
    return p, models, log


def weekly(p: pd.DataFrame, value: str, relevant: str | None, weeks: pd.DatetimeIndex):
    """Pilot method: paragraph -> document mean -> weekly mean of documents -> carry forward."""
    sub = p[p[value].notna()]
    if relevant:
        sub = sub[sub[relevant].fillna(False).astype(bool)]
    docs = sub.groupby("doc_id").agg(date=("date", "first"), v=(value, "mean"))
    raw = docs.set_index("date")["v"].resample("W-FRI").mean()
    filled = raw.reindex(raw.index.union(weeks)).ffill().reindex(weeks)
    return filled, raw.reindex(weeks).notna()


def build(p, models, weeks):
    out = pd.DataFrame(index=weeks)
    out.index.name = "Date"
    ms = p[p.doc_type.isin(["minutes", "statement"])]
    for m in models:
        rel, pu, eu = f"{m}_policy_relevant", f"{m}_policy_uncertainty", f"{m}_economic_uncertainty"
        out[f"{m}_policy"], out[f"{m}_new_policy_text"] = weekly(p, pu, rel, weeks)
        out[f"{m}_policy_ms"], _ = weekly(ms, pu, rel, weeks)
        out[f"{m}_economic"], _ = weekly(p, eu, None, weeks)
        share = p[p[rel].notna()].assign(r=lambda d: d[rel].astype(float))
        out[f"{m}_policy_share"] = share.set_index("date")["r"].resample("W-FRI").mean().reindex(weeks)
    out["dictionary"], _ = weekly(p, "dict_uncertainty", "dict_policy_relevant", weeks)
    wk = p.groupby("doc_id").agg(date=("date", "first")).set_index("date").assign(n=1)["n"]
    out["n_documents"] = wk.resample("W-FRI").sum().reindex(weeks).fillna(0).astype(int)
    out["new_fed_text"] = out["n_documents"] > 0
    return out


def kappa(a, b):
    a, b = a.astype(bool), b.astype(bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a, b in RECESSIONS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f1f0ec", zorder=0, lw=0)


def fig_by_model(w, models):
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.7 * n + 0.6), sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    top = max(w[f"{m}_policy"].max() for m in models) + 0.3
    for ax, (m, (label, color)) in zip(axes, models.items()):
        style(ax)
        ax.plot(w.index, w[f"{m}_policy"], color=color, alpha=0.28, lw=0.8)
        ax.plot(w.index, w[f"{m}_policy"].rolling(13, min_periods=4).mean(), color=color, lw=2)
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, int(top) + 1, 2))
        ax.set_title(label, loc="left", fontsize=10.5, color=INK, pad=4)
        ax.set_ylabel("uncertainty (0-10)", fontsize=8.5, color=INK2)
        for day, text in EVENTS.items():
            ax.axvline(pd.Timestamp(day), color=GRID, lw=0.9, zorder=0)
    # Event labels above the top panel, staggered on two rows so neighbours don't collide.
    for i, (day, text) in enumerate(EVENTS.items()):
        axes[0].annotate(text.replace("\n", " "), (pd.Timestamp(day), 1), xycoords=("data", "axes fraction"),
                         xytext=(0, 3 + 10 * (i % 3)), textcoords="offset points", ha="center", va="bottom",
                         fontsize=6.5, color=INK2, annotation_clip=False)
    axes[0].set_title(axes[0].get_title(loc="left"), loc="left", fontsize=10.5, color=INK, pad=38)
    axes[-1].set_xlim(w.index[0], w.index[-1])
    fig.text(0.01, 0.005, "Faint line: weekly index. Thick line: 13-week trailing average. Shaded: NBER "
             "recessions. Speeches enter in 1996, press conferences in April 2011.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUTPUT / "fig_index_by_model.png", dpi=220)
    plt.close(fig)


def fig_compared(w, models):
    """Two panels on one scale: the main index of every model, then the minutes + statements check."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, suffix, title in zip(axes, ("policy", "policy_ms"),
                                 ("Main index (all Fed text), 13-week averages",
                                  "Check: FOMC minutes and statements only, 13-week averages")):
        style(ax)
        for m, (label, color) in models.items():
            ax.plot(w.index, w[f"{m}_{suffix}"].rolling(13, min_periods=4).mean(), color=color, lw=1.8, label=label)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4)
        ax.set_ylabel("uncertainty (0-10)", fontsize=8.5, color=INK2)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=len(models), frameon=False, fontsize=8)
    axes[-1].set_xlim(w.index[0], w.index[-1])
    fig.tight_layout()
    fig.savefig(OUTPUT / "fig_models_compared.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="use whatever is scored so far (testing only)")
    args = ap.parse_args()

    p, models, log = load(args.partial)
    weeks = pd.DatetimeIndex(pd.read_csv(RAW / "weekly_dates_1993_2015.csv")["Date"])
    assert len(weeks) == 1118 and (weeks.day_of_week == 4).all()
    w = build(p, models, weeks)
    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    say("=== Scores: coverage, errors, tokens")
    for r in log:
        say(f"{r['model']}: {r['scored']:,} paragraphs scored ({r['coverage']:.1%}), error rows {r['error_rows']} "
            f"{r['errors'] or ''}; tokens in {r['in_tok']:,} / out {r['out_tok']:,}")
    say()
    say("=== Share of paragraphs marked 'about future policy', by document type")
    share = pd.DataFrame({m: p.groupby("doc_type")[f"{m}_policy_relevant"].mean() for m in models}).round(3)
    share.loc["all"] = [p[f"{m}_policy_relevant"].mean().round(3) for m in models]
    say(share.to_string())
    say()
    say("=== Policy-uncertainty score on policy paragraphs: mean (sd) by document type")
    for m in models:
        rel = p[p[f"{m}_policy_relevant"].fillna(False).astype(bool)]
        g = rel.groupby("doc_type")[f"{m}_policy_uncertainty"].agg(["mean", "std", "count"]).round(2)
        say(f"{m}:\n{g.to_string()}")
    say()
    say("=== Model agreement (paragraph level)")
    names = list(models)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = p[p[f"{a}_policy_relevant"].notna() & p[f"{b}_policy_relevant"].notna()]
            k = kappa(both[f"{a}_policy_relevant"], both[f"{b}_policy_relevant"])
            pol = both[both[f"{a}_policy_relevant"].astype(bool) & both[f"{b}_policy_relevant"].astype(bool)]
            gap = (pol[f"{a}_policy_uncertainty"] - pol[f"{b}_policy_uncertainty"]).abs()
            say(f"{a} vs {b}: same yes/no {np.mean(both[f'{a}_policy_relevant'] == both[f'{b}_policy_relevant']):.1%}, "
                f"kappa {k:.2f}; on {len(pol):,} paragraphs both call policy: within 1 point {np.mean(gap <= 1):.0%}, "
                f"within 2 {np.mean(gap <= 2):.0%}, mean gap {gap.mean():.2f}")
    say()
    say("=== Weekly index: correlation between models (levels / week-to-week changes / 13-week averages)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = w[f"{a}_policy"], w[f"{b}_policy"]
            say(f"{a} vs {b}: {x.corr(y):.2f} / {x.diff().corr(y.diff()):.2f} / "
                f"{x.rolling(13).mean().corr(y.rolling(13).mean()):.2f}")
    for m in names:
        say(f"{m}: main index vs minutes+statements-only check, 13-week averages: "
            f"{w[f'{m}_policy'].rolling(13).mean().corr(w[f'{m}_policy_ms'].rolling(13).mean()):.2f}")
    say()
    say("=== Weeks with new policy text (the rest carry the last value forward)")
    say(pd.DataFrame({m: w[f"{m}_new_policy_text"].groupby(w.index.year).mean() for m in names}).round(2).T.to_string())
    say()
    say("=== Known Fed history: average index in each period (expectation written BEFORE looking)")
    rows = []
    for a, b, name, expect in HISTORY:
        sl = w.loc[a:b]
        rows.append({"period": f"{a[:7]} to {b[:7]}", "episode": name, "expected": expect,
                     **{m: round(sl[f"{m}_policy"].mean(), 2) for m in names},
                     **{f"{m}_ms": round(sl[f"{m}_policy_ms"].mean(), 2) for m in names}})
    hist = pd.DataFrame(rows)
    say(hist.to_string(index=False))
    say(f"full-sample means: " + ", ".join(f"{m} {w[f'{m}_policy'].mean():.2f}" for m in names))
    say()

    # ---- save
    cols = ["new_fed_text", "n_documents"]
    for m in names:
        cols += [f"{m}_policy", f"{m}_policy_ms", f"{m}_economic", f"{m}_policy_share", f"{m}_new_policy_text"]
    cols += ["dictionary"]
    out = w[cols].round(4)
    out.index = out.index.date
    out.index.name = "Date"
    out.to_csv(OUTPUT / "fed_uncertainty_weekly_1994_2015.csv")

    notes = pd.DataFrame({"column": ["Date", "new_fed_text", "n_documents"] + [c for m in names for c in (
        f"{m}_policy", f"{m}_policy_ms", f"{m}_economic", f"{m}_policy_share", f"{m}_new_policy_text")] + ["dictionary"],
        "meaning": ["Friday ending the week (the weekly Friday dates, 1993-12-31 to 2015-05-29)",
                    "True if the Fed released any document that week",
                    "Number of Fed documents released that week"] + [d for m in names for d in (
            f"MAIN INDEX ({models[m][0]}): how uncertain the Fed sounds about its future policy path, 0-10. "
            "Average over paragraphs the model marks as about future policy; document mean, then weekly mean; "
            "weeks with no new policy text keep the previous value.",
            f"Check version ({models[m][0]}): same, using only FOMC minutes and statements (available all years).",
            f"Economic uncertainty ({models[m][0]}), 0-10, averaged over ALL paragraphs that week (blank = no text).",
            f"Share of that week's paragraphs the model marks as about future policy (blank = no text).",
            f"True if the week had new policy text for this model (False = value carried forward).")] +
        ["No-AI benchmark: uncertainty words per 100 words in policy paragraphs (Loughran-McDonald based)."]})
    cov = pd.read_csv(PROCESSED / "coverage_by_year.csv")
    with pd.ExcelWriter(OUTPUT / "fed_uncertainty_weekly_1994_2015.xlsx") as xl:
        out.to_excel(xl, sheet_name="weekly_index")
        notes.to_excel(xl, sheet_name="notes", index=False)
        cov.to_excel(xl, sheet_name="documents_per_year", index=False)
        hist.to_excel(xl, sheet_name="known_history_check", index=False)

    fig_by_model(w, models)
    fig_compared(w, models)
    (OUTPUT / "checks.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
