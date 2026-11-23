"""Shared settings for the pilot. Change START_YEAR / END_YEAR to rerun the whole pipeline
for a different period (2021 or later: earlier years sit on older Fed pages with another layout)."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Which run the scripts work on. Default is the 2023-2026 pilot. Set ECON_RUN=hist to work on
# the historical sample instead (weekly, Friday 1993-12-31 to Friday 2015-05-29); it has
# its own processed/ and output/ folders, so the pilot's files are never touched.
RUN = os.environ.get("ECON_RUN", "pilot")
if RUN == "hist":
    START_YEAR, END_YEAR = 1993, 2015
    TODAY = date(2015, 5, 29)          # last week in the sample; nothing after it is used
    FIRST_WEEK = date(1993, 12, 31)    # first week in the sample
    PROCESSED = ROOT / "data" / "processed_1994_2015"
    OUTPUT = ROOT / "output" / "1994_2015"
else:
    START_YEAR, END_YEAR = 2021, 2026
    TODAY = date.today()
    FIRST_WEEK = date(START_YEAR, 1, 1)
    PROCESSED = ROOT / "data" / "processed"
    OUTPUT = ROOT / "output"
YEARS = range(START_YEAR, END_YEAR + 1)
PERIOD = f"{START_YEAR}-{END_YEAR}"

FED = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (undergraduate thesis research; python-requests)"}

for p in (RAW, PROCESSED, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)
