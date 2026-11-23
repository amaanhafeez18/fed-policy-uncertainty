"""Step 1, historical run: Fed communication for the historical sample (weekly, Dec 31, 1993 to May 29, 2015).

Run with ECON_RUN=hist so everything lands in data/processed_1994_2015/ and the pilot is untouched:
    ECON_RUN=hist .venv/bin/python src/01h_collect_fed_text_historical.py

Same four document types and the same rules as the pilot's step 1 (01_collect_fed_text.py): FOMC
statements, FOMC minutes (dated when RELEASED), press conferences (Chair's answers are kept later, in
step 3) and Board speeches. The pilot reads the Fed's JSON feeds, but those only start in 2006, so
this script reads the Fed's older archive pages instead:

  Meetings, statements, minutes,     /monetarypolicy/fomchistoricalYYYY.htm (one page per year, every
  press conferences                  meeting and conference call with links to what was published)
  Minutes release dates  2006-2015   press-release feed /json/ne-press.json (as in the pilot)
                         1996-2005   press-release archive /newsevents/press/all/YYYYall.htm
                         1993-1995   not listed online. Until 2004 the minutes came out a few days
                                     after the NEXT meeting; we use the Friday after the next meeting,
                                     the pattern the 1996 releases follow (checked and printed below).
  Speeches               2006-2015   speech feed /json/ne-speeches.json (as in the pilot)
                         1996-2005   /newsevents/speech/YYYYspeech.htm. None are online before 1996.

What exists changes over the sample, and the coverage table printed at the end shows it:
statements only after policy changes until May 1999, speeches from 1996, press conferences from
April 2011. Fed transcripts and Greenbooks are NOT used: they were released 5 years later, so
markets couldn't see them at the time.

Clean-up: the same speech is sometimes listed on several dates (repeat deliveries, same text): only
the first listing is kept. Some old Fed pages contain the character U+FFFD where an apostrophe was;
it is put back as an apostrophe.

Output: documents.csv and coverage_by_year.csv in the historical processed folder.
"""
import io
import json
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import FED, HEADERS, PROCESSED, RAW, RUN

assert RUN == "hist", "run with ECON_RUN=hist so the pilot's files aren't overwritten"

FIRST_RELEASE = datetime(1993, 11, 1)   # a little before the first week, so it starts with a value
LAST_RELEASE = datetime(2015, 5, 29)    # the last week
CACHE = RAW / "fed_pages"               # shared with the pilot's step 1: nothing is downloaded twice
CACHE.mkdir(exist_ok=True)


def fetch(path: str) -> bytes:
    """Download once, then reuse the saved copy. Same cache naming as 01_collect_fed_text.py."""
    local = CACHE / re.sub(r"[^A-Za-z0-9._-]", "_", path.strip("/"))
    if local.exists():
        return local.read_bytes()
    for attempt in range(4):
        try:
            r = requests.get(FED + path, headers=HEADERS, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.ConnectionError(f"status {r.status_code}")
            r.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    local.write_bytes(r.content)
    time.sleep(0.5)
    return r.content


def page_text(path: str) -> str:
    return _page_text(path).replace("\ufffd", "'")


def _page_text(path: str) -> str:
    """Text of an HTML page, one paragraph per block, separated by blank lines.

    Pages in the Fed's current layout (an #article element) are read exactly as in the pilot: one
    paragraph per <p>. Older pages (1990s and early 2000s) often put text outside <p> tags, e.g. the
    policy directive inside a <blockquote>, so there every block-level element starts a new paragraph."""
    soup = BeautifulSoup(fetch(path), "lxml")
    article = soup.select_one("#article")
    if article is not None:
        for tag in article.select("script, style, .shareDL, .lastUpdate, .footnotes"):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in article.find_all("p")]
        return "\n\n".join(p for p in paras if p)
    body = soup.body
    for tag in body.select("script, style, noscript"):
        tag.decompose()
    # Line breaks in the HTML source are not paragraph breaks: many 1990s-2000s pages put a blank line after
    # every source line, even mid-paragraph. Collapse them first (the first version did not, so lines under
    # 12 words were later dropped and paragraphs came out scrambled; fixed in the verification pass).
    for s in list(body.find_all(string=True)):
        if s.find_parent("pre") is None:
            s.replace_with(re.sub(r"\s+", " ", str(s)))
    sep = " "  # paragraph separator: only block-level elements and double <br> start a new paragraph
    for tag in body.find_all(["p", "div", "blockquote", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5",
                              "table", "tr", "td", "center", "pre", "hr", "dl", "dd", "dt"]):
        tag.insert_before(sep)
        tag.insert_after(sep)
    for br in body.find_all("br"):
        br.replace_with("\n")
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(rf"{sep}|\n\s*\n", body.get_text())]
    return "\n\n".join(p for p in paras if p)


def extra_text(path: str) -> str:
    """Text that page_text() misses on pages in the Fed's current layout: blocks with no <p> inside them
    (quote blocks, bullet points, table cells, headings). In the 2012-2015 minutes this is where the policy
    directive, the statement the Committee voted on and the 2014 normalization principles sit.

    Kept apart from the main text (column extra_text) so the paragraphs already scored keep their IDs;
    step 3 turns it into extra paragraphs with their own IDs. Blocks under 12 words (buttons, labels) are
    skipped, as step 3 would drop them anyway. Older pages and PDFs return "": nothing is missed there."""
    if path.lower().endswith(".pdf"):
        return ""
    article = BeautifulSoup(fetch(path), "lxml").select_one("#article")
    if article is None:
        return ""
    for tag in article.select("script, style, .shareDL, .lastUpdate, .footnotes"):
        tag.decompose()
    blocks = []
    for el in article.find_all(["blockquote", "li", "td", "h2", "h3", "h4", "h5", "div"]):
        # innermost text blocks only, and nothing page_text() already took from a <p>
        if el.find_parent("p") or el.find("p") or el.find(["blockquote", "li", "div", "table"]):
            continue
        text = el.get_text(" ", strip=True)
        if len(text.split()) >= 12:
            blocks.append(text)
    return "\n\n".join(blocks).replace("�", "'")


def pdf_text(path: str) -> str:
    reader = PdfReader(io.BytesIO(fetch(path)))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def try_get(kind: str, func, path: str) -> str | None:
    try:
        return func(path)
    except requests.HTTPError as e:
        print(f"    skipped {kind} {path} ({e.response.status_code})")
        return None


def feed(name: str) -> list[dict]:
    return json.loads(fetch(f"/json/{name}.json").decode("utf-8-sig"))


def mdy(s: str) -> datetime:
    return datetime.strptime(s.split()[0], "%m/%d/%Y")


def last8(href: str) -> str:
    return re.findall(r"(\d{8})", href)[-1]


def in_window(d: datetime) -> bool:
    return FIRST_RELEASE <= d <= LAST_RELEASE


# ---------------------------------------------------------------- FOMC meetings
def meeting_links() -> tuple[list[dict], list[str]]:
    """Statements and press conferences from the yearly history pages, plus every meeting with minutes."""
    statements, minutes, pdf_minutes = [], set(), {}
    for year in range(1993, 2016):
        soup = BeautifulSoup(fetch(f"/monetarypolicy/fomchistorical{year}.htm"), "lxml")
        for panel in soup.select(".panel"):
            heading = panel.select_one(".panel-heading")
            heading = heading.get_text(" ", strip=True) if heading else ""
            for a in panel.find_all("a"):
                href, label = a.get("href") or "", a.get_text(" ", strip=True)
                href = href.replace(FED, "")
                if label == "Statement" and re.search(r"\d{8}", href):
                    statements.append(dict(kind="statement", date=last8(href), href=href, heading=heading))
                elif label == "Press Conference" and "presconf" in href:
                    d = last8(href)
                    statements.append(dict(kind="press_conference", date=d,
                                           href=f"/mediacenter/files/FOMCpresconf{d}.pdf", heading=heading))
                elif "minutes" in href.lower() and href.lower().endswith(".htm") and re.search(r"\d{8}", href):
                    minutes.add((last8(href), href))
                elif "minutes" in href.lower() and href.lower().endswith(".pdf") and re.search(r"\d{8}", href):
                    pdf_minutes[last8(href)] = href
    # Meetings whose minutes are only posted as a PDF (June 24-25, 2008): read the PDF.
    have = [datetime.strptime(d, "%Y%m%d") for d, _ in minutes]
    minutes |= {(d, h) for d, h in pdf_minutes.items()  # (PDF and HTML can be named a day apart)
                if all(abs((datetime.strptime(d, "%Y%m%d") - x).days) > 2 for x in have)}
    return statements, sorted(minutes)


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
MINUTES_TITLE = re.compile(r"^Minutes of (the )?Federal Open Market Committee,?\s*(.*)$")


def meeting_key(dates: str) -> str | None:
    """Last meeting day named in a minutes title, as YYYYMMDD (the minutes file is named after it).
    Handles 'June 28-29, 2006', 'July 31-August 1, 2012', 'December 6 and 11, 2007' and
    'September 29, October 7, and October 28-29, 2008'."""
    years = re.findall(r"\d{4}", dates)
    months = list(re.finditer(MONTHS, dates))
    if not years or not months:
        return None
    tail = dates[months[-1].end(): dates.rfind(years[-1])]
    days = re.findall(r"\d{1,2}", tail)
    if not days:
        return None
    return datetime.strptime(f"{months[-1].group(0)} {days[-1]} {years[-1]}", "%B %d %Y").strftime("%Y%m%d")


def all_keys(dates: str) -> list[str]:
    """Every 'Month D, YYYY' or 'Month D-D, YYYY' in a title, as first-day and last-day keys."""
    out = []
    for mo, d1, d2, yr in re.findall(rf"({MONTHS}) (\d{{1,2}})(?:[–-](\d{{1,2}}))?, (\d{{4}})", dates):
        for d in (d1, d2 or d1):
            out.append(datetime.strptime(f"{mo} {d} {yr}", "%B %d %Y").strftime("%Y%m%d"))
    return out


def minutes_release_dates(meetings: list[str]) -> tuple[dict[str, datetime], dict[str, str]]:
    """meeting date (YYYYMMDD) -> release date, and how each date was found."""
    released, source = {}, {}
    # 2006 onward: the press-release feed, as in the pilot.
    undated = []
    for item in feed("ne-press"):
        m = MINUTES_TITLE.match(item.get("t", ""))
        if not m or not item.get("d"):
            continue
        key = meeting_key(m.group(2))
        if key:
            released[key], source[key] = mdy(item["d"]), "press-release feed"
        else:  # e.g. May 22, 2013: "Minutes of the Federal Open Market Committee" with no meeting date
            undated.append(mdy(item["d"]))
    for rel in undated:  # belongs to the last meeting 2-5 weeks before it that has no release yet
        cands = [k for k in meetings if 14 <= (rel - datetime.strptime(k, "%Y%m%d")).days <= 35]
        if cands and cands[-1] not in released:
            released[cands[-1]], source[cands[-1]] = rel, "press-release feed (title without date)"
    # 1996-2005: the Board's yearly press-release archive pages (release date, then the title).
    for year in range(1996, 2006):
        soup = BeautifulSoup(fetch(f"/newsevents/press/all/{year}all.htm"), "lxml")
        for li in soup.find_all("li"):
            a = li.find("a")
            m = MINUTES_TITLE.match(a.get_text(" ", strip=True)) if a else None
            date_text = li.find(string=True, recursive=False)
            if not m or not date_text or not date_text.strip():
                continue
            # A title can name two meetings ("August 21, 2001, and September 17, 2001"): register each.
            keys = {meeting_key(m.group(2))} | set(all_keys(m.group(2)))
            for key in keys - {None}:
                if key not in released:
                    released[key] = datetime.strptime(date_text.strip(), "%B %d, %Y")
                    source[key] = "press-release archive"
    # The yearly history pages name 1990s two-day meetings after the FIRST day (19960130), the
    # release titles after the last (January 30-31 -> 19960131): match those one or two days apart.
    for k in meetings:
        if k not in released:
            d = datetime.strptime(k, "%Y%m%d")
            for gap in (1, 2):
                alt = (d + timedelta(days=gap)).strftime("%Y%m%d")
                if alt in released and alt not in meetings:
                    released[k], source[k] = released[alt], source[alt]
                    break
    # Not listed anywhere: use the release practice of the time.
    for this, nxt in zip(meetings, meetings[1:] + [None]):
        if this in released:
            continue
        d = datetime.strptime(this, "%Y%m%d")
        if this >= "20041214":  # since Dec 2004 the minutes come out three weeks after the meeting
            released[this], source[this] = d + timedelta(days=21), "rule: three weeks after meeting"
        elif nxt is not None:   # before that, a few days after the next meeting
            nd = datetime.strptime(nxt, "%Y%m%d")
            released[this] = nd + timedelta(days=(4 - nd.weekday()) % 7 or 7)
            source[this] = "rule: Friday after next meeting"
    return released, source


def check_rule(meetings: list[str], released: dict, source: dict) -> None:
    """How well does 'Friday after the next meeting' match the real release dates of 1996-1999?"""
    hits, total = 0, 0
    for this, nxt in zip(meetings, meetings[1:]):
        if source.get(this) != "press-release archive" or not this.startswith(("1996", "1997", "1998", "1999")):
            continue
        nd = datetime.strptime(nxt, "%Y%m%d")
        guess = nd + timedelta(days=(4 - nd.weekday()) % 7 or 7)
        total += 1
        hits += abs((guess - released[this]).days) <= 3
    print(f"  check of the 1993-95 rule on 1996-99 minutes: within 3 days of the real date "
          f"(same week) in {hits} of {total}")


# ---------------------------------------------------------------- speeches
def old_speech_list() -> list[dict]:
    out = []
    for year in range(1996, 2006):
        soup = BeautifulSoup(fetch(f"/newsevents/speech/{year}speech.htm"), "lxml")
        for li in soup.find_all("li"):
            a = li.select_one(".title a")
            date_text = li.find(string=True, recursive=False)
            if not a or not date_text or not date_text.strip():
                continue
            try:
                d = datetime.strptime(date_text.strip(), "%B %d, %Y")
            except ValueError:
                continue
            speaker = li.select_one(".speaker")
            out.append(dict(date=d, href=a["href"].replace(FED, ""), title=a.get_text(" ", strip=True),
                            speaker=speaker.get_text(" ", strip=True) if speaker else ""))
    return out


def speech_slug(href: str) -> str:
    parts = [p for p in href.rstrip("/").split("/") if p]
    last = parts[-1].removesuffix(".htm")
    return parts[-2] if last in ("default", "index") else last


def main():
    rows = []
    statements, minutes = meeting_links()

    # Statements and press conferences
    for s in statements:
        day = datetime.strptime(s["date"], "%Y%m%d")
        if not in_window(day):
            continue
        if s["kind"] == "statement":
            text = try_get("statement", page_text, s["href"])
            doc_id = f"{s['date']}-statement"  # named after the Fed's file, kept stable so scores stay linked
            if text:
                # Date the statement by the release date printed in it. The file name is usually the same day,
                # but the Fed's 2007 page links the June 28, 2007 statement as .../20070618a.htm.
                printed = re.search(rf"({MONTHS}) (\d{{1,2}}), (\d{{4}})", text[:300])
                if printed:
                    printed = datetime.strptime(" ".join(printed.groups()), "%B %d %Y")
                    if abs((printed - day).days) > 1:
                        print(f"    statement {s['href']}: file name says {day:%Y-%m-%d}, text says "
                              f"{printed:%Y-%m-%d}; using the date in the text")
                        day = printed
                rows.append(dict(doc_id=doc_id, date=day, doc_type="statement", speaker="FOMC",
                                 title=f"FOMC statement {day:%Y-%m-%d} ({s['heading']})", url=FED + s["href"],
                                 text=text))
        else:
            text = try_get("press conference", pdf_text, s["href"])
            if text:
                rows.append(dict(doc_id=f"{s['date']}-presconf", date=day, doc_type="press_conference",
                                 speaker="Chair", title=f"Press conference transcript {s['date']}",
                                 url=FED + s["href"], text=text))
    print(f"statements: {sum(r['doc_type'] == 'statement' for r in rows)}, "
          f"press conferences: {sum(r['doc_type'] == 'press_conference' for r in rows)}")

    # Minutes, dated by release
    meetings = [m for m, _ in minutes]
    released, source = minutes_release_dates(meetings)
    check_rule(meetings, released, source)
    n = len(rows)
    for mdate, href in minutes:
        rel = released.get(mdate)
        if rel is None:
            print(f"    no release date for minutes {mdate}; skipped")
            continue
        if not in_window(rel):
            continue
        text = try_get("minutes", pdf_text if href.lower().endswith(".pdf") else page_text, href)
        if text:
            rows.append(dict(doc_id=f"{rel:%Y%m%d}-minutes{mdate}", date=rel, doc_type="minutes", speaker="FOMC",
                             title=f"Minutes of meeting {mdate} (released {rel:%Y-%m-%d}; {source[mdate]})",
                             url=FED + href, text=text))
    print(f"minutes: {len(rows) - n}")

    # Speeches
    speeches = old_speech_list()
    for s in feed("ne-speeches"):
        if s.get("d") and s.get("l"):
            speeches.append(dict(date=mdy(s["d"]), href=s["l"], title=s.get("t", ""), speaker=s.get("s", "")))
    speeches = [s for s in speeches if in_window(s["date"])]
    first = {}
    for s in sorted(speeches, key=lambda s: s["date"]):
        first.setdefault(s["href"], s)
    print(f"speeches listed: {len(speeches)}; dropped {len(speeches) - len(first)} repeat listings of the same text")
    speeches = sorted(first.values(), key=lambda s: s["date"])
    n = len(rows)
    for i, s in enumerate(speeches, 1):
        text = try_get("speech", page_text, s["href"])
        if text:
            rows.append(dict(doc_id=f"{s['date']:%Y%m%d}-speech-{speech_slug(s['href'])}", date=s["date"],
                             doc_type="speech", speaker=s["speaker"], title=s["title"], url=FED + s["href"],
                             text=text))
        if i % 100 == 0:
            print(f"    {i}/{len(speeches)} speeches")
    print(f"speeches: {len(rows) - n}")

    df = pd.DataFrame(rows).sort_values(["date", "doc_id"]).reset_index(drop=True)
    dup = df["doc_id"].duplicated(keep=False)
    if dup.any():  # e.g. two statements on one day: number them
        df.loc[dup, "doc_id"] += "-" + df[dup].groupby("doc_id").cumcount().add(1).astype(str)
    assert df["doc_id"].is_unique
    df["n_words"] = df["text"].str.split().str.len()
    df["extra_text"] = df["url"].map(lambda u: extra_text(u.replace(FED, "")))
    df["n_words_extra"] = df["extra_text"].str.split().str.len().fillna(0).astype(int)
    print(f"extra text outside <p> on current-layout pages: {df.n_words_extra.sum():,} words "
          f"in {(df.n_words_extra > 0).sum()} documents")
    df.to_csv(PROCESSED / "documents.csv", index=False)

    cov = df.assign(year=df["date"].dt.year).pivot_table(index="year", columns="doc_type", values="doc_id",
                                                        aggfunc="count", fill_value=0)
    cov.to_csv(PROCESSED / "coverage_by_year.csv")
    print(f"\nSaved {len(df)} documents to {PROCESSED / 'documents.csv'}")
    print(cov)
    print(df.groupby("doc_type")["n_words"].agg(["count", "sum", "mean", "min"]).round(0))


if __name__ == "__main__":
    main()
