"""Step 3: cut each document into passages of roughly one paragraph (60-250 words).

Why passages and not whole documents: an LLM judges a paragraph far more reliably than a
40-page set of minutes, and averaging many passage scores gives a smoother, more defensible
index. It also lets us hand-check individual passages later.

Cleaning rules (each one is a choice you should be able to justify):
  - Press conferences: keep only the Chair's answers. Reporters' questions are not Fed
    communication, so they are dropped.
  - Minutes: drop the attendance list (lines that are mostly names and titles).
  - All documents: drop boilerplate (voting records, media contacts, footnote links).
Output: data/processed/passages.csv
"""
import re

import pandas as pd

from config import PROCESSED, RUN

MIN_WORDS, MAX_WORDS = 60, 250
BOILERPLATE = re.compile(
    r"^(Voting (for|against)|For media inquiries|Implementation Note|Return to text|"
    r"Last Update|Board of Governors|For release at|Share$|Accessible Keys|Related Information|"
    r"A (joint )?meeting of the Federal Open Market Committee|Attendance|"
    r"\d{1,3}\.\s)",  # footnotes: "1. These views are my own...", "3. See Waller..."
    re.I,
)
# Historical run (ECON_RUN=hist): numbered paragraphs are KEPT. The pilot's footnote rule above also
# dropped numbered points in the body of speeches ("5. Improve corporate governance...") and the
# numbered parts of the minutes, about 220,000 words of Fed text in 1994-2015. Footnotes are Fed text
# too, so they are scored like everything else (the models mark citations as not about policy).
KEEP_NUMBERED = RUN == "hist"
BOILERPLATE_HIST = re.compile(BOILERPLATE.pattern.replace(r"|\d{1,3}\.\s)", ")"), re.I)
SPEAKER_TURN = re.compile(r"(?m)^\s*([A-Z][A-Z.’'\- ]{3,40})\.\s")


def looks_like_names(p: str) -> bool:
    """Attendance lists and reference lists: lots of commas, mostly capitalized words."""
    words = p.split()
    caps = sum(w[:1].isupper() for w in words) / max(len(words), 1)
    return caps > 0.6 or (p.count(",") >= 3 and caps > 0.45)


def clean_paragraphs(text: str) -> list[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    if KEEP_NUMBERED:
        paras = [re.sub(r"\s*Return to text\s*$", "", p) for p in paras]  # footnote link, not text
        rule = BOILERPLATE_HIST
    else:
        rule = BOILERPLATE
    return [p for p in paras if len(p.split()) >= 12 and not rule.match(p) and not looks_like_names(p)]


def chair_turns(text: str) -> list[str]:
    """Split a press-conference transcript into speaker turns and keep the Chair's."""
    text = re.sub(r"(?m)^.*Press Conference\s+FINAL\s*$", "", text)  # page headers
    text = re.sub(r"(?m)^\s*Page \d+ of \d+\s*$", "", text)
    parts = SPEAKER_TURN.split(text)
    # parts = [preamble, SPEAKER1, words1, SPEAKER2, words2, ...]
    return [re.sub(r"\s+", " ", words).strip()
            for speaker, words in zip(parts[1::2], parts[2::2]) if speaker.startswith("CHAIR")]


def split_long(p: str) -> list[str]:
    """Break text longer than MAX_WORDS at sentence boundaries."""
    if len(p.split()) <= MAX_WORDS:
        return [p]
    out, cur = [], []
    for sent in re.split(r"(?<=[.?!])\s+", p):
        cur.append(sent)
        if len(" ".join(cur).split()) >= MAX_WORDS * 0.6:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def merge_short(pieces: list[str]) -> list[str]:
    """Glue consecutive short paragraphs together until each passage has MIN_WORDS."""
    out, buf = [], ""
    for p in pieces:
        buf = f"{buf} {p}".strip()
        if len(buf.split()) >= MIN_WORDS:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf.split()) < MIN_WORDS / 2:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return out


def main():
    docs = pd.read_csv(PROCESSED / "documents.csv", parse_dates=["date"])
    rows = []
    for d in docs.itertuples():
        paras = chair_turns(d.text) if d.doc_type == "press_conference" else clean_paragraphs(d.text)
        pieces = [s for p in paras for s in split_long(p)]
        for i, passage in enumerate(merge_short(pieces)):
            rows.append(dict(passage_id=f"{d.doc_id}-{i:03d}", doc_id=d.doc_id, date=d.date,
                             doc_type=d.doc_type, speaker=d.speaker, title=d.title,
                             text=passage, n_words=len(passage.split())))
        # Historical run: text on current-layout pages that sits outside <p> (quote blocks, bullet points;
        # see extra_text() in 01h). Separate IDs ("-x000") so existing paragraphs keep theirs, same doc_id so
        # they count towards the same document in the index.
        extra = getattr(d, "extra_text", None)
        if KEEP_NUMBERED and isinstance(extra, str) and extra.strip():
            pieces = [s for p in clean_paragraphs(extra) for s in split_long(p)]
            for i, passage in enumerate(merge_short(pieces)):
                rows.append(dict(passage_id=f"{d.doc_id}-x{i:03d}", doc_id=d.doc_id, date=d.date,
                                 doc_type=d.doc_type, speaker=d.speaker, title=d.title,
                                 text=passage, n_words=len(passage.split())))
    out = pd.DataFrame(rows)
    assert out["passage_id"].is_unique
    out.to_csv(PROCESSED / "passages.csv", index=False)
    print(f"Saved {len(out)} passages to {PROCESSED / 'passages.csv'}")
    print(out.groupby("doc_type")["n_words"].agg(["count", "mean", "sum"]).round(0))


if __name__ == "__main__":
    main()
