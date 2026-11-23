"""Sanity check of the historical documents (run after 01h_collect_fed_text_historical.py).

    ECON_RUN=hist .venv/bin/python src/01h_check_documents.py

Prints what was collected and flags anything that looks wrong: empty or tiny documents, error or
redirect pages saved as text, duplicates, dates outside the sample, minutes released before their
meeting, web-page junk, broken characters. Writes documents_flags.csv (one row per flagged
document) next to documents.csv.
"""
import hashlib
import re

import pandas as pd

from config import PROCESSED, RAW, RUN

assert RUN == "hist"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

df = pd.read_csv(PROCESSED / "documents.csv", parse_dates=["date"])
df["year"] = df["date"].dt.year
weeks = pd.to_datetime(pd.read_csv(RAW / "weekly_dates_1993_2015.csv")["Date"])
flags = []


def flag(rows, why):
    for r in rows.itertuples():
        flags.append(dict(doc_id=r.doc_id, doc_type=r.doc_type, date=r.date.date(), n_words=r.n_words,
                          problem=why, url=r.url))
    return len(rows)


print(f"=== {len(df)} documents, released {df.date.min().date()} to {df.date.max().date()}\n")

print("--- Documents per year and type")
cov = df.pivot_table(index="year", columns="doc_type", values="doc_id", aggfunc="count", fill_value=0)
cov["total"] = cov.sum(axis=1)
print(cov.to_string(), "\n")

print("--- Words per document")
print(df.groupby("doc_type")["n_words"].describe(percentiles=[.05, .5, .95]).round(0).to_string(), "\n")

print("--- Checks (number of documents flagged)")
min_words = {"statement": 40, "minutes": 1500, "speech": 400, "press_conference": 2000}
tiny = df[df.n_words < df.doc_type.map(min_words)]
print(f"suspiciously short (statement<40, minutes<1500, speech<400, press conf<2000 words): {flag(tiny, 'short')}")

junk = re.compile(r"Page not Found|Skip to main content|http-equiv|window\.__CF|Just a moment|Access Denied|"
                  r"enable JavaScript|cookies", re.I)
print(f"error / redirect / web junk in text: {flag(df[df.text.str.contains(junk)], 'web junk')}")

broken = re.compile(r"â€|Ã©|Ã¨|�")
print(f"broken characters (encoding): {flag(df[df.text.str.contains(broken)], 'encoding')}")

h = df.text.map(lambda t: hashlib.md5(t.encode()).hexdigest())
print(f"exact duplicate texts: {flag(df[h.duplicated(keep=False)], 'duplicate text')}")
print(f"duplicate URLs: {flag(df[df.url.duplicated(keep=False)], 'duplicate url')}")

out = df[(df.date < "1993-11-01") | (df.date > "2015-05-29")]
print(f"released outside Nov 1, 1993 - May 29, 2015: {flag(out, 'outside window')}")

m = df[df.doc_type == "minutes"].copy()
m["meeting"] = pd.to_datetime(m.doc_id.str.extract(r"minutes(\d{8})")[0])
m["lag"] = (m.date - m.meeting).dt.days
print(f"minutes released before or on the meeting day: {flag(m[m.lag <= 0], 'minutes before meeting')}")
print(f"minutes released more than 9 weeks after the meeting: {flag(m[m.lag > 63], 'minutes very late')}")

expect = {"statement": r"Committee|federal funds|Federal Reserve", "minutes": r"Federal Open Market Committee",
          "press_conference": r"CHAIR", "speech": r"\w"}
miss = df[~df.apply(lambda r: bool(re.search(expect[r.doc_type], r.text)), axis=1)]
print(f"missing the words expected for their type: {flag(miss, 'unexpected content')}")

wk = df[df.date.dt.day_of_week >= 5]
print(f"dated on a weekend: {flag(wk, 'weekend date')}\n")

print("--- Minutes: release lag after the meeting (days), by how the date was found")
m["source"] = m.title.str.extract(r"; (.*)\)$")[0]
print(m.groupby("source")["lag"].describe()[["count", "min", "50%", "max"]].to_string(), "\n")

print("--- Statements: meetings vs. statements per year (statements were only issued after policy")
print("    changes until May 1999; extra ones are unscheduled moves between meetings)")
mins_per_year = m.groupby(m.meeting.dt.year).size()
st = df[df.doc_type == "statement"].groupby("year").size()
print(pd.DataFrame({"meetings_with_minutes": mins_per_year, "statements": st}).fillna(0).astype(int).T.to_string(), "\n")

print("--- Press conferences")
print(", ".join(df[df.doc_type == "press_conference"].date.dt.strftime("%Y-%m-%d")), "\n")

print("--- Speeches: who spoke (top 12)")
sp = df[df.doc_type == "speech"]
print(sp.speaker.str.replace(r"^(Chairman|Chair|Vice Chairman|Vice Chair|Governor)\s+", "", regex=True)
      .value_counts().head(12).to_string(), "\n")

print("--- The 1,118 weeks: how many have new Fed text?")
wk_of = df.date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
has = pd.Series(weeks.isin(set(wk_of)).values, index=weeks.values)
by_year = has.groupby(has.index.year).agg(["sum", "count"])
by_year["share_with_new_text"] = (by_year["sum"] / by_year["count"]).round(2)
print(by_year.rename(columns={"sum": "weeks_with_text", "count": "weeks"}).T.to_string())
print(f"overall: {has.mean():.0%} of weeks have new text; the rest carry the previous value forward\n")

f = pd.DataFrame(flags)
f.to_csv(PROCESSED / "documents_flags.csv", index=False)
print(f"--- {len(f)} flags written to documents_flags.csv")
if len(f):
    print(f.groupby(["problem", "doc_type"]).size().to_string())
