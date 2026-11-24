"""Every number shown on the web page and in the PDF report, computed in one place.

    import site_stats; S = site_stats.compute()      (run with ECON_RUN=hist)

Inputs are the pipeline's saved outputs: documents.csv, passages.csv, scores_llm_*.csv, the weekly
index (output/1994_2015/fed_uncertainty_weekly_1994_2015.csv) and the comparison series in data/raw.
"""
import importlib
import re

import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT, PROCESSED, RAW, RUN

assert RUN == "hist"
MODELS = {"luna": ("gpt-6-luna", "GPT-6 Luna", "OpenAI", "low reasoning effort"),
          "gemini": ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "Google", "default thinking"),
          "haiku": ("claude-haiku-5-5", "Claude Haiku 5.5", "Anthropic", "low reasoning effort")}
TYPES = [("statement", "FOMC statements"), ("minutes", "FOMC minutes"), ("speech", "Board speeches"),
         ("press_conference", "Press conferences")]
MEASURES = [  # id, label, short label, frequency, group
    ("hrs", "Husted–Rogers–Sun: Fed policy uncertainty in newspapers", "Husted–Rogers–Sun", "monthly", "uncertainty"),
    ("bbd", "Baker–Bloom–Davis: monetary policy uncertainty", "Baker–Bloom–Davis", "monthly", "uncertainty"),
    ("vix", "VIX: expected stock-market volatility", "VIX", "weekly", "uncertainty"),
    ("vol2", "2-year Treasury yield volatility", "2-year yield volatility", "weekly", "uncertainty"),
    ("cpi", "CPI inflation, year on year", "CPI inflation", "monthly", "macro"),
    ("unemp", "Unemployment rate", "Unemployment rate", "monthly", "macro")]
RECESSIONS = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30")]
EVENTS = [("1994-02-04", "First hike, Feb 1994"), ("1998-09-29", "LTCM cuts"), ("2001-01-03", "Easing begins"),
          ("2001-09-17", "9/11"), ("2003-08-12", '"Considerable period"'), ("2004-06-30", '"Measured pace"'),
          ("2007-08-17", "Credit crunch"), ("2008-09-16", "Lehman"), ("2008-12-16", "Zero bound"),
          ("2011-08-09", '"Mid-2013"'), ("2013-05-22", "Taper tantrum"), ("2014-12-17", '"Patient"')]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fred(s):
    d = pd.read_csv(RAW / "hist_fred" / f"fred_{s}.csv")
    d.columns = ["date", s]
    d["date"] = pd.to_datetime(d["date"])
    d[s] = pd.to_numeric(d[s], errors="coerce")
    return d.set_index("date")[s].dropna()


def nice_title(doc_type, title, date):
    """A reader-facing title: minutes, statements and press conferences by date; speeches keep their own title."""
    long = lambda t: f"{t:%B} {t.day}, {t.year}"
    if doc_type == "minutes":
        meeting = re.match(r"Minutes of meeting (\d{8})", title)
        assert meeting, title
        return f"Minutes of the FOMC meeting of {long(pd.Timestamp(meeting.group(1)))}" + (" (release date estimated)" if "rule:" in title else "")
    if doc_type == "statement":
        return f"FOMC statement, {long(pd.Timestamp(date))}"
    if doc_type == "press_conference":
        return f"Press conference, {long(pd.Timestamp(date))}"
    return title


def crit(n):
    t = stats.t.ppf(0.975, n - 2)
    return float(t / np.sqrt(n - 2 + t ** 2))


def _corr(x, y):
    ok = x.notna() & y.notna()
    return float(x[ok].corr(y[ok])), int(ok.sum())


def _kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def load():
    p = pd.read_csv(PROCESSED / "passages.csv", parse_dates=["date"])
    for m, (tag, *_rest) in MODELS.items():
        s = pd.read_csv(PROCESSED / f"scores_llm_{tag}.csv").drop_duplicates("passage_id", keep="last")
        s = s[["passage_id", "policy_relevant", "policy_uncertainty", "economic_uncertainty", "stance", "reason"]]
        p = p.merge(s.rename(columns={c: f"{m}_{c}" for c in s.columns if c != "passage_id"}), on="passage_id", how="left")
    assert p[[f"{m}_policy_relevant" for m in MODELS]].notna().all().all(), "every paragraph needs every model's answer"
    for m in MODELS:
        p[f"{m}_policy_relevant"] = p[f"{m}_policy_relevant"].astype(bool)
    docs = pd.read_csv(PROCESSED / "documents.csv", parse_dates=["date"])
    w = pd.read_csv(OUTPUT / "fed_uncertainty_weekly_1994_2015.csv", index_col=0, parse_dates=True)
    return p, docs, w


def comparison_series(weeks):
    vix = _fred("VIXCLS").resample("W-FRI").mean().reindex(weeks)
    vol2 = (_fred("DGS2").diff() * 100).resample("W-FRI").std().reindex(weeks)
    hrs = pd.read_excel(RAW / "hrs_mpu_monthly.xlsx").dropna()
    hrs = hrs[hrs["month"].astype(str).str.match(r"^\d{4}m\d{1,2}$")]
    hrs.index = pd.to_datetime(hrs["month"].str.replace("m", "-") + "-01")
    epu = pd.read_excel(RAW / "categorical_epu.xlsx")
    epu = epu[pd.to_numeric(epu["Year"], errors="coerce").notna()]
    epu.index = pd.to_datetime(epu["Year"].astype(int).astype(str) + "-" + epu["Month"].astype(int).astype(str) + "-01")
    months = pd.date_range("1993-12-01", "2015-05-01", freq="MS")
    monthly = pd.DataFrame({"hrs": pd.to_numeric(hrs["US MPU"], errors="coerce"),
                            "bbd": pd.to_numeric(epu["2. Monetary policy"], errors="coerce"),
                            "cpi": _fred("CPIAUCSL").pct_change(12) * 100,
                            "unemp": _fred("UNRATE")}).reindex(months)
    weekly = pd.DataFrame({"vix": vix, "vol2": vol2}, index=weeks)
    return weekly, monthly


def compute():
    p, docs, w = load()
    r18 = importlib.import_module("18_historical_results")
    m3 = importlib.import_module("03_make_passages")
    S = {}
    S["n_documents"], S["n_paragraphs"], S["n_weeks"] = len(docs), len(p), len(w)
    S["first_week"], S["last_week"] = str(w.index[0].date()), str(w.index[-1].date())

    # documents by type and year
    rows = []
    for t, label in TYPES:
        d = docs[docs.doc_type == t]
        first = d.date.min()
        rows.append({"doc_type": t, "label": label, "first": f"{MON[first.month - 1]} {first.year}", "documents": len(d),
                     "paragraphs": int((p.doc_type == t).sum()),
                     "words": int(d.n_words.sum() + d.get("n_words_extra", pd.Series(0, index=d.index)).sum())})
    S["doc_types"] = rows
    S["docs_per_year"] = [{"year": int(y), **{t: int(((docs.date.dt.year == y) & (docs.doc_type == t)).sum()) for t, _ in TYPES}}
                          for y in range(1993, 2016)]

    # cleaning: words kept and removed, by kind
    removed = {"Lists of names (attendance, vote tallies, bibliographies)": 0,
               "Short lines (headings, datelines, venue lines, page navigation)": 0,
               "Reporters' questions (press conferences)": 0,
               "Fixed headers ('A meeting of the FOMC was held...', 'For release at...')": 0}
    keys = list(removed)
    for r in docs.itertuples():
        if r.doc_type == "press_conference":
            removed[keys[2]] += r.n_words - sum(len(t.split()) for t in m3.chair_turns(r.text))
            continue
        for t in [r.text] + ([r.extra_text] if isinstance(getattr(r, "extra_text", None), str) else []):
            for para in [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n\s*\n", t)]:
                para = re.sub(r"\s*Return to text\s*$", "", para)
                n = len(para.split())
                if not n:
                    continue
                if n < 12:
                    removed[keys[1]] += n
                elif m3.BOILERPLATE_HIST.match(para):
                    removed[keys[3]] += n
                elif m3.looks_like_names(para):
                    removed[keys[0]] += n
    total = int(docs.n_words.sum() + docs.get("n_words_extra", pd.Series(0)).sum())
    kept = int(p.n_words.sum())
    clean = [("Kept in paragraphs that the models scored", kept)] + [(k, int(v)) for k, v in removed.items()]
    rest = total - sum(v for _, v in clean)
    assert 0 <= rest < 0.005 * total
    clean.append(("Footnote links ('Return to text')", int(rest)))
    S["cleaning"] = [{"category": k, "words": v, "share_pct": 100 * v / total} for k, v in clean]
    S["kept_pct"] = 100 * kept / total

    # models
    episodes = r18.HISTORY
    ep_rows = []
    for a, b, name, expect in episodes:
        sl = w.loc[a:b]
        da, db_ = pd.Timestamp(a), pd.Timestamp(b)
        ep_rows.append({"period": f"{MON[da.month - 1]} {da.year} – {MON[db_.month - 1]} {db_.year}", "episode": name,
                        "expected": expect, **{m: float(sl[f"{m}_policy"].mean()) for m in MODELS}})
    for m in MODELS:
        vals = [r[m] for r in ep_rows]
        for r in ep_rows:
            r[f"{m}_top"], r[f"{m}_bottom"] = r[m] == max(vals), r[m] == min(vals)
    S["history"] = ep_rows
    models = []
    for m, (tag, label, company, setting) in MODELS.items():
        rel = p[f"{m}_policy_relevant"]
        sc = p.loc[rel, f"{m}_policy_uncertainty"]
        ranked = sorted(ep_rows, key=lambda r: -r[m])
        by_type = []
        for t, tl in TYPES:
            sub = p[p.doc_type == t]
            pol = sub[sub[f"{m}_policy_relevant"]]
            by_type.append({"label": tl, "share_pct": 100 * len(pol) / len(sub), "mean_score": float(pol[f"{m}_policy_uncertainty"].mean()), "n": len(pol)})
        models.append({"id": m, "label": label, "company": company, "setting": setting,
                       "policy_share_pct": 100 * rel.mean(), "policy_paragraphs": int(rel.sum()), "mean_score": float(sc.mean()),
                       "weekly_mean": float(w[f"{m}_policy"].mean()), "weekly_sd": float(w[f"{m}_policy"].std()),
                       "top_episode": ranked[0]["episode"], "bottom_episode": ranked[-1]["episode"], "by_type": by_type})
    S["models"] = models

    # agreement
    agr = []
    roll = lambda s: s.rolling(13).mean()
    for a, b, label in [("luna", "gemini", "Luna and Gemini"), ("luna", "haiku", "Luna and Haiku"), ("gemini", "haiku", "Gemini and Haiku")]:
        ra, rb = p[f"{a}_policy_relevant"], p[f"{b}_policy_relevant"]
        both = ra & rb
        gap = (p.loc[both, f"{a}_policy_uncertainty"] - p.loc[both, f"{b}_policy_uncertainty"]).abs()
        agr.append({"pair": f"{a}-{b}", "label": label, "same_pct": 100 * (ra == rb).mean(), "kappa": _kappa(ra, rb),
                    "within1_pct": 100 * (gap <= 1).mean(), "r_weekly": float(w[f"{a}_policy"].corr(w[f"{b}_policy"])),
                    "r13": float(roll(w[f"{a}_policy"]).corr(roll(w[f"{b}_policy"]))),
                    "r13_ms": float(roll(w[f"{a}_policy_ms"]).corr(roll(w[f"{b}_policy_ms"])))})
    S["agreement"] = agr
    S["speech_mean"] = {m["id"]: next(x["mean_score"] for x in m["by_type"] if x["label"] == "Board speeches") for m in models}

    # how much text each version of the index rests on
    years = (w.index[-1] - w.index[0]).days / 365.25
    ms_docs = docs[docs.doc_type.isin(["minutes", "statement"])]
    new_text = lambda d: (pd.Series(1, index=d.date).resample("W-FRI").sum().reindex(w.index).fillna(0) > 0).mean()
    S["coverage"] = {"all_docs_per_year": len(docs) / years, "ms_docs_per_year": len(ms_docs) / years,
                     "all_weeks_pct": 100 * new_text(docs), "ms_weeks_pct": 100 * new_text(ms_docs),
                     "ms_paragraph_pct": 100 * p.doc_type.isin(["minutes", "statement"]).mean()}

    # comparisons with other measures
    weekly, monthly = comparison_series(w.index)
    mi = w[[f"{m}_policy" for m in MODELS]].resample("MS").mean()
    mi.columns = list(MODELS)
    monthly = monthly.reindex(mi.index)
    corr_rows, series = [], {}
    for mid, label, short, freq, group in MEASURES:
        if freq == "weekly":
            X, Y, dates, k = w[[f"{m}_policy" for m in MODELS]].set_axis(list(MODELS), axis=1), weekly[mid], w.index, 13
        else:
            X, Y, dates, k = mi, monthly[mid], mi.index, 3
        series[mid] = {"label": label, "short": short, "freq": freq, "dates": [d.strftime("%Y-%m-%d") for d in dates], "series": {}}
        for m in MODELS:
            lv, n = _corr(X[m], Y)
            ch, nc = _corr(X[m].diff(), Y.diff())
            c = crit(nc)
            corr_rows.append({"key": f"{mid}|{m}", "measure": label, "short": short, "model": MODELS[m][1], "freq": freq, "group": group,
                              "level": lv, "change": ch, "crit": c, "n": n, "sig_change": abs(ch) > c, "sig_level": abs(lv) > crit(n)})
            ok = X[m].notna() & Y.notna()
            z = lambda s: ((s - s[ok].mean()) / s[ok].std()).where(ok)
            sm = lambda s: s.rolling(k).mean()
            r3 = lambda s: [None if pd.isna(v) else round(float(v), 3) for v in s]
            series[mid]["series"][m] = {"model": r3(sm(z(X[m]))), "measure": r3(sm(z(Y))), "level": lv, "change": ch, "crit": c, "n": n}
    S["correlations"], S["measure_series"] = corr_rows, series
    u = [r for r in corr_rows if r["group"] == "uncertainty"]
    S["n_comparisons"] = len(u)
    S["max_change"] = max(abs(r["change"]) for r in u)
    S["n_sig"] = sum(r["sig_change"] for r in u)
    # the same tests for the minutes-and-statements version (changes only)
    mi_ms = w[[f"{m}_policy_ms" for m in MODELS]].resample("MS").mean().set_axis(list(MODELS), axis=1)
    ms_rows = []
    for mid, label, short, freq, group in MEASURES:
        if group != "uncertainty":
            continue
        X, Y = (w[[f"{m}_policy_ms" for m in MODELS]].set_axis(list(MODELS), axis=1), weekly[mid]) if freq == "weekly" else (mi_ms, monthly[mid].reindex(mi_ms.index))
        for m in MODELS:
            ch, nc = _corr(X[m].diff(), Y.diff())
            ms_rows.append({"measure": short, "model": m, "change": ch, "sig": abs(ch) > crit(nc)})
    S["ms_max_change"] = max(abs(r["change"]) for r in ms_rows)
    S["ms_n_sig"] = sum(r["sig"] for r in ms_rows)
    S["ms_correlations"] = ms_rows
    S["weekly"], S["monthly_benchmarks"], S["passages"], S["docs"], S["w"] = weekly, monthly, p, docs, w
    return S
