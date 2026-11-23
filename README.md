# How uncertain does the Fed sound? A weekly index read by three AI models, 1994–2015

**Read it here: https://amaanhafeez18.github.io/fed-policy-uncertainty/** (interactive, with a light and dark mode)
· [PDF report](report/Fed_policy_uncertainty_index_1994-2015.pdf)

A weekly index of how uncertain the Federal Reserve sounds about its own future policy path. Three AI language models,
GPT-6 Luna (OpenAI), Gemini 3.1 Flash-Lite (Google) and Claude Haiku 5.5 (Anthropic), each scored 40,039
paragraphs from 1,518 Federal Reserve documents (every FOMC statement, set of minutes and press conference, and
every Board speech, released from November 1993 to May 2015), with no date or speaker shown. The scores are averaged into
a weekly series on 1,118 Fridays, December 31, 1993 to May 29, 2015, and compared with established measures of
uncertainty and with the economy.

## Contents

| Path | What it is |
|---|---|
| `index.html`, `assets/` | The web page |
| `report/` | The same content as a PDF |
| `data/weekly_index_1993_2015.csv` / `.xlsx` | The weekly index for each model |
| `data/paragraphs_scored.csv.gz` | Every scored paragraph with each model's answer (scores, stance, reason) |
| `data/documents.csv.gz` | Every Fed document: release date, type, speaker, link and full text |
| `data/comparison_*.csv` | Comparison series (Husted–Rogers–Sun, Baker–Bloom–Davis, CPI, unemployment, 2-year yield volatility) |
| `data/correlations.csv`, `model_agreement.csv` | The tables on the page |
| `data/prompt.txt` | The instructions every model received |
| `code/` | The pipeline: collection, cleaning, scoring, index, tables, charts, page and PDF |

## Sources

- Federal Reserve Board, federalreserve.gov: statements, minutes, press conference transcripts, speeches.
- FRED, Federal Reserve Bank of St. Louis: 2-year Treasury yield (DGS2), CPI (CPIAUCSL), unemployment rate (UNRATE),
  VIX (VIXCLS). VIX values are not redistributed here.
- Husted, Rogers and Sun, "Monetary Policy Uncertainty" (US MPU index).
- Baker, Bloom and Davis, "Measuring Economic Policy Uncertainty", policyuncertainty.com (monetary policy category).
- NBER business cycle dates (recession shading).

Amaan Hafeez, Western University.
