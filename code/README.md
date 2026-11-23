# Code

Python 3 with pandas, numpy, scipy, matplotlib, reportlab, openpyxl, requests, beautifulsoup4, lxml, pypdf, d3 (in the page),
and the openai, google-genai and anthropic client libraries for scoring. Run every step with `ECON_RUN=hist` set, from this
folder, so outputs go to `data/processed_1994_2015/` and `output/1994_2015/`.

| Step | Script |
|---|---|
| Collect every Fed document (Nov 1993 – May 2015) | `src/01h_collect_fed_text_historical.py` |
| Check the documents | `src/01h_check_documents.py` |
| Cut into paragraphs and clean | `src/03_make_passages.py` |
| Word-count benchmark | `src/04a_score_dictionary.py` |
| Score with each model (needs API keys: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY) | `src/04b_score_openai.py`, `src/04b_score_gemini.py`, `src/04b_score_claude.py --model claude-haiku-5-5` |
| Weekly index | `src/05_build_index.py`, `src/18_historical_results.py` |
| Comparison with other measures | `src/21_validate_historical.py` |
| Every number on the page and in the PDF | `src/site_stats.py` |
| PDF report | `src/24_make_report_pdf.py` |

The instructions given to every model are in `src/rubric.py`. Comparison data are in `data/raw/`; VIX is not
redistributed, so download VIXCLS from FRED to `data/raw/hist_fred/fred_VIXCLS.csv` before step 21.
