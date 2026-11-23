"""PDF version of the web page: same sections, same numbers (site_stats.compute()), static charts.

    ECON_RUN=hist .venv/bin/python src/24_make_report_pdf.py

Writes site/report/Fed_policy_uncertainty_index_1994-2015.pdf (and its charts to output/1994_2015/report_figs/).
"""
import re
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

import site_stats
from config import OUTPUT, ROOT
from rubric import SYSTEM_PROMPT

OUT = ROOT / "site" / "report" / "Fed_policy_uncertainty_index_1994-2015.pdf"
FIG = OUTPUT / "report_figs"
FIG.mkdir(parents=True, exist_ok=True)
SITE_URL = "https://amaanhafeez18.github.io/fed-policy-uncertainty/"
REPO = "https://github.com/amaanhafeez18/fed-policy-uncertainty"

C = {"luna": "#1f63d6", "gemini": "#cf5410", "haiku": "#12865a", "measure": "#3a404c",
     "statement": "#0f7c8a", "minutes": "#5a6ea8", "speech": "#c48a12", "press_conference": "#9a4fb6"}
INK, MUTED, LINE, GRID, BAND = "#15171c", "#555c69", "#dfe2e8", "#e9ebf0", "#e8eaee"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 220, "font.family": "DejaVu Sans"})

ink, muted, line_c = (colors.HexColor(c) for c in (INK, MUTED, LINE))
accent, panel = colors.HexColor("#1c5fc4"), colors.HexColor("#f4f5f7")
S_ = {
    "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=ink, spaceAfter=4),
    "by": ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=13, textColor=muted, spaceAfter=10),
    "lead": ParagraphStyle("l", fontName="Helvetica", fontSize=11, leading=15.5, textColor=ink, spaceAfter=8),
    "eye": ParagraphStyle("e", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=muted, spaceBefore=14),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=ink, spaceBefore=2, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=ink, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=9.8, leading=14, textColor=ink, spaceAfter=6),
    "note": ParagraphStyle("n", fontName="Helvetica", fontSize=8.6, leading=12, textColor=muted, spaceAfter=6),
    "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.8, leading=11.2, textColor=ink),
    "cellb": ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.8, leading=11.2, textColor=ink),
    "head": ParagraphStyle("hd", fontName="Helvetica-Bold", fontSize=8.2, leading=10.5, textColor=muted),
    "mono": ParagraphStyle("m", fontName="Courier", fontSize=7.6, leading=10, textColor=ink),
}
f2 = lambda v: f"{v:.2f}"
pct = lambda v: f"{v:.1f}%"
sg = lambda v: ("+" if v >= 0 else "-") + f"{abs(v):.2f}"
num = lambda v: f"{v:,}"
esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(rows, widths, right=(), bold=None):
    data = []
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            st = S_["head"] if i == 0 else (S_["cellb"] if bold and bold(i, j) else S_["cell"])
            if j in right:
                st = ParagraphStyle("r", parent=st, alignment=2)
            raw = re.search(r"</?(b|u|i|font)\b", str(c))
            cells.append(Paragraph(str(c) if raw else esc(c), st))
        data.append(cells)
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1.2, line_c), ("LINEBELOW", (0, 1), (-1, -1), 0.5, line_c),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def shade(ax):
    for a, b in site_stats.RECESSIONS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=BAND, lw=0, zorder=0)


def datefmt(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(length=0)


def fig_docs(S):
    rows = pd.DataFrame(S["docs_per_year"]).set_index("year")
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bottom = np.zeros(len(rows))
    for t, label in [("minutes", "Minutes"), ("statement", "Statements"), ("speech", "Speeches"), ("press_conference", "Press conferences")]:
        ax.bar(rows.index, rows[t], bottom=bottom, color=C[t], width=0.78, label=label)
        bottom += rows[t].values
    ax.set_xticks(range(1993, 2016, 2))
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Documents")
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.16), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "docs.png")
    plt.close(fig)


def fig_model(S, m):
    w = S["w"]
    s = w[f"{m}_policy"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    shade(ax)
    for d, label in site_stats.EVENTS:
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.6, ls=(0, (2, 3)), alpha=0.5, zorder=1)
    ax.plot(w.index, s, color=C[m], lw=0.8, alpha=0.35, label="Weekly index")
    ax.plot(w.index, s.rolling(13).mean(), color=C[m], lw=2.2, label="13-week average")
    for k, (d, label) in enumerate(site_stats.EVENTS):
        ax.text(pd.Timestamp(d), 8.2 + 0.62 * (k % 3), label, fontsize=6.8, color=MUTED, ha="center", va="bottom")
    ax.set_ylim(0, 8)
    ax.set_xlim(w.index[0], w.index[-1])
    ax.set_ylabel("Policy uncertainty (0-10)")
    datefmt(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.11), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / f"model_{m}.png")
    plt.close(fig)


def fig_compare(S):
    w = S["w"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.1), sharex=True, sharey=True)
    for ax, suf, title in zip(axes, ("_policy", "_policy_ms"), ("All Fed text (the main index), 13-week averages",
                                                               "FOMC minutes and statements only, 13-week averages")):
        shade(ax)
        for m, (tag, label, *_r) in site_stats.MODELS.items():
            ax.plot(w.index, w[m + suf].rolling(13).mean(), color=C[m], lw=1.8, label=label)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(1, 5.5)
        datefmt(ax)
    axes[0].legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32), fontsize=9)
    axes[-1].set_xlim(w.index[0], w.index[-1])
    fig.tight_layout()
    fig.savefig(FIG / "compare.png")
    plt.close(fig)


def fig_measures(S, ids, name):
    fig, axes = plt.subplots(len(ids), 3, figsize=(7.2, 2.05 * len(ids) + 0.5), sharey=True)
    for r, mid in enumerate(ids):
        ms = S["measure_series"][mid]
        dates = pd.to_datetime(ms["dates"])
        for c, m in enumerate(site_stats.MODELS):
            ax = axes[r, c]
            s = ms["series"][m]
            ax.plot(dates, [np.nan if v is None else v for v in s["measure"]], color=C["measure"], lw=1.3, ls=(0, (4, 3)))
            ax.plot(dates, [np.nan if v is None else v for v in s["model"]], color=C[m], lw=1.6)
            ax.set_title(f"{site_stats.MODELS[m][1]}\nlevels {sg(s['level'])}, changes {sg(s['change'])}", fontsize=8, loc="left", color=INK)
            ax.xaxis.set_major_locator(mdates.YearLocator(7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(length=0, labelsize=7.5)
            ax.set_ylim(-3, 3.5)
            if c == 0:
                ax.set_ylabel(ms["short"], fontsize=8.5, color=INK)
    fig.text(0.01, 0.003, "Dashed: the measure. Solid: the model. Standard deviations from each series' own average, smoothed "
             "13 weeks or 3 months.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def img(path, width=7.2):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(str(path), width=width * inch, height=width * inch * h / w)


def main():
    S = site_stats.compute()
    fig_docs(S)
    for m in site_stats.MODELS:
        fig_model(S, m)
    fig_compare(S)
    fig_measures(S, ["hrs", "bbd", "vix"], "measures_a")
    fig_measures(S, ["vol2", "cpi", "unemp"], "measures_b")

    st = [Paragraph("How uncertain does the Fed sound? A weekly index read by three AI models, 1994-2015", S_["title"]),
          Paragraph(f"Amaan Hafeez · Western University · {date.today():%B %Y}<br/>Interactive version: {SITE_URL}<br/>"
                    f"Every data file and the code: {REPO}", S_["by"]),
          Paragraph(f"Three AI language models scored <b>{num(S['n_paragraphs'])}</b> paragraphs from <b>{num(S['n_documents'])}</b> "
                    "Federal Reserve documents for how uncertain the Fed sounds about its own policy path. They largely agree on "
                    "which paragraphs are about policy, but their weekly indices barely move with established uncertainty measures: "
                    f"across <b>{S['n_comparisons']}</b> comparisons with four such measures, the largest correlation of changes is "
                    f"<b>{f2(S['max_change'])}</b>, and <b>{S['n_sig']}</b> pass the 5% cut-off.", S_["lead"])]

    data_part = [Paragraph("4 · THE DATA", S_["eye"]), Paragraph("Every FOMC statement, set of minutes and press conference, and every Board speech", S_["h"]),
           Paragraph("All text is from federalreserve.gov, dated by the day it became public. Minutes count from their release: about "
                     "three weeks after the meeting from December 2004, a few days after the next meeting before that.", S_["p"]),
           table([["Document type", "First in the sample", "Documents", "Paragraphs scored", "Words"]] +
                 [[r["label"], r["first"], num(r["documents"]), num(r["paragraphs"]), num(r["words"])] for r in S["doc_types"]],
                 [2.0, 1.4, 1.0, 1.3, 1.2], right=(2, 3, 4)),
           Paragraph("Documents released each year", S_["h3"]), img(FIG / "docs.png"),
           Paragraph("Before May 1999 the Fed issued a statement only when it changed policy, so 1994-95 rest mostly on the minutes.", S_["note"])]

    steps = ["<b>Read each page:</b> the document's own text only, without menus, page headers or links.",
             "<b>Cut into paragraphs</b> at the Fed's own breaks, roughly 60 to 250 words each: short pieces join the next one, long ones "
             "split at a sentence end.",
             "<b>Remove what is not the Fed speaking:</b> attendance lists, vote tallies, bibliographies, headings, datelines, and "
             "reporters' questions at press conferences.",
             "<b>Keep everything else,</b> including footnotes, numbered points, bullet lists and the policy directive quoted in the minutes."]
    cleaning_part = [Paragraph("5 · CLEANING", S_["eye"]), Paragraph("How the text was cleaned", S_["h"]),
           ListFlowable([ListItem(Paragraph(s, S_["p"]), leftIndent=12) for s in steps], bulletType="1", leftIndent=14, bulletFontSize=9.5),
           Paragraph(f"<b>{pct(S['kept_pct'])}</b> of all words end up in a paragraph the models scored.", S_["p"])]

    fields = [("policy_relevant", "Yes or no: is the paragraph about the future path of U.S. monetary policy (the federal funds rate, "
               "the balance sheet, or how the Fed will decide)?"),
              ("policy_uncertainty", "0 to 10: how uncertain the Fed sounds about its own future policy path; 0 if not about policy. "
               "<b>The index is built from this number.</b>"),
              ("economic_uncertainty", "0 to 10: how uncertain it sounds about the economic outlook."),
              ("stance", "Hawkish, neutral or dovish, kept separate so a firm commitment is not read as certainty or uncertainty."),
              ("reason", "At most 25 words, quoting the phrase behind the score.")]
    scale = [("0", "A definite, pre-committed path"), ("2-3", "A clear direction with mild conditioning"),
             ("5", "Direction known; pace or endpoint open and data dependent"), ("7-8", "The path is highly uncertain; several scenarios possible"),
             ("10", "The Fed says it does not know what it will do")]
    sc = Table([[Paragraph(f"<font size=13><b>{a}</b></font><br/><font color='{MUTED}' size=7.8>{b}</font>", S_["cell"]) for a, b in scale]],
               colWidths=[1.44 * inch] * 5)
    sc.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, line_c), ("INNERGRID", (0, 0), (-1, -1), 0.6, line_c),
                            ("BACKGROUND", (0, 0), (-1, -1), panel), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    prompt_part = [Paragraph("6 · THE PROMPT AND THE MODELS", S_["eye"]), Paragraph("What each model was asked", S_["h"]),
           Paragraph("Each paragraph goes to each model on its own, with no date, speaker or document name. All three get the same "
                     "instructions (in full at the end) and answer five fields:", S_["p"]),
           table([["Field", "Meaning"]] + [[f"<font name='Courier'>{a}</font>", b] for a, b in fields], [1.6, 5.6]),
           Spacer(1, 8), sc, Spacer(1, 8),
           Paragraph("<b>From paragraphs to weeks:</b> each model's scores on the paragraphs it calls about policy are averaged within each "
                     "document, then across the documents released each week (weeks end on Friday). A week with no new policy text keeps "
                     "the previous value.", S_["p"])]

    models_part = [Paragraph("The three models", S_["h3"]),
           Paragraph("Three low-cost models from different companies, to see whether the result depends on which model reads the text.", S_["p"]),
           table([["Model", "Company and setting", "Paragraphs about policy", "Average score on those", "Weekly index, average"]] +
                 [[m["label"], f"{m['company']}, {m['setting']}", pct(m["policy_share_pct"]), f2(m["mean_score"]), f2(m["weekly_mean"])] for m in S["models"]],
                 [1.6, 1.9, 1.3, 1.2, 1.2], right=(2, 3, 4)),
           Paragraph("How much they agree", S_["h3"]),
           table([["Models", "Same yes/no on 'about policy'", "Kappa", "Scores within 1 point", "Weekly index corr.", "13-week averages", "13-week, minutes + statements"]] +
                 [[r["label"], pct(r["same_pct"]), f2(r["kappa"]), pct(r["within1_pct"]), f2(r["r_weekly"]), f2(r["r13"]), f2(r["r13_ms"])] for r in S["agreement"]],
                 [1.3, 1.1, 0.6, 1.0, 0.95, 0.95, 1.3], right=(1, 2, 3, 4, 5, 6)),
           Paragraph("Kappa is agreement beyond chance (0 is chance, 1 is perfect). 'Scores within 1 point' counts paragraphs both models "
                     "call about policy.", S_["note"])]

    each_part = []
    for i, m in enumerate(S["models"]):
        block = [Paragraph("2 · EACH MODEL" if i == 0 else " ", S_["eye"]),
                 Paragraph(f"{m['label']} <font color='{MUTED}' size=10>({m['company']}, {m['setting']})</font>", S_["h"]),
                 img(FIG / f"model_{m['id']}.png"),
                 Paragraph("Faint line: weekly index. Thick line: 13-week average. Shaded: NBER recessions. Dotted lines: events labelled above.", S_["note"])]
        bullets = [f"Weekly index: average <b>{f2(m['weekly_mean'])}</b> on the 0-10 scale, standard deviation <b>{f2(m['weekly_sd'])}</b>.",
                   f"Calls <b>{pct(m['policy_share_pct'])}</b> of all paragraphs about future policy, with an average score of <b>{f2(m['mean_score'])}</b> on those."]
        block += [ListFlowable([ListItem(Paragraph(b, S_["p"]), leftIndent=10) for b in bullets], bulletType="bullet", leftIndent=12, start="•")]
        each_part += ([PageBreak()] + block) if i == 0 else [KeepTogether(block)]

    agr = {r["pair"]: r for r in S["agreement"]}
    cov = S["coverage"]
    span = lambda v: f"{min(v):.2f} to {max(v):.2f}"
    full = dict(Jan="January", Feb="February", Mar="March", Apr="April", May="May", Jun="June", Jul="July", Aug="August",
                Sep="September", Oct="October", Nov="November", Dec="December")
    first = {r["doc_type"]: full[r["first"][:3]] + r["first"][3:] for r in S["doc_types"]}
    side_part = [Paragraph("1 · THE INDEX", S_["eye"]), Paragraph("The weekly index: three models together", S_["h"]), img(FIG / "compare.png"),
           Paragraph(f"In the main index Luna and Haiku track each other (<b>{f2(agr['luna-haiku']['r13'])}</b>); Gemini runs higher and moves "
                     f"differently (<b>{f2(agr['luna-gemini']['r13'])}</b> with Luna, <b>{f2(agr['gemini-haiku']['r13'])}</b> with Haiku), mostly "
                     "because it reads speeches as more uncertain: its average score on speech paragraphs about policy is "
                     f"<b>{f2(S['speech_mean']['gemini'])}</b>, against <b>{f2(S['speech_mean']['luna'])}</b> for Luna and "
                     f"<b>{f2(S['speech_mean']['haiku'])}</b> for Haiku.", S_["p"]),
           Paragraph(f"<b>Why the minutes-and-statements version matters.</b> Minutes and statements are the only documents published in every "
                     f"year of the sample (speeches start in {first['speech']}, press conferences in {first['press_conference']}). Holding the "
                     "mix of text fixed shows whether the main index moves only because new kinds of documents enter it. The models agree "
                     f"more evenly here ({span([r['r13_ms'] for r in S['agreement']])} between pairs, against {span([r['r13'] for r in S['agreement']])} "
                     f"on all text), but the index rests on far less text: about {cov['ms_docs_per_year']:.0f} documents a year instead of "
                     f"{cov['all_docs_per_year']:.0f}, new text in {cov['ms_weeks_pct']:.0f}% of weeks instead of {cov['all_weeks_pct']:.0f}%, and "
                     f"{cov['ms_paragraph_pct']:.0f}% of the paragraphs. It is a check, not a better index: its largest correlation of changes "
                     f"with the four uncertainty measures is {f2(S['ms_max_change'])}, with {S['ms_n_sig']} of {S['n_comparisons']} beyond the 5% "
                     "cut-off.", S_["p"])]

    rows = [["Measure", "Model", "Levels", "Changes", "5% cut-off", "Obs."]]
    sig = set()
    for k, r in enumerate(S["correlations"], start=1):
        rows.append([r["short"], r["model"], sg(r["level"]), sg(r["change"]), f2(r["crit"]), num(r["n"])])
        if r["sig_level"]:
            sig.add((k, 2))
        if r["sig_change"]:
            sig.add((k, 3))
    measures_part = [PageBreak(), Paragraph("3 · AGAINST OTHER MEASURES", S_["eye"]), Paragraph("Each model against established measures and the economy", S_["h"]),
           Paragraph("An index of Fed policy uncertainty should move with other measures of it: the Husted-Rogers-Sun newspaper index of "
                     "uncertainty about Fed policy, the Baker-Bloom-Davis monetary-policy index, the VIX and the volatility of the 2-year "
                     "Treasury yield. Inflation and unemployment show how it relates to the economy. Levels ask whether the index is high when "
                     "the measure is high; changes ask whether it rises when the measure rises. Changes are the stricter test, because series "
                     "that trend over 21 years correlate in levels even when unrelated.", S_["p"]),
           img(FIG / "measures_a.png"), img(FIG / "measures_b.png"),
           Paragraph("All correlations", S_["h3"]),
           table(rows, [2.2, 1.7, 0.8, 0.8, 0.9, 0.7], right=(2, 3, 4, 5), bold=lambda i, j: (i, j) in sig),
           Paragraph("Bold: beyond the 5% cut-off, which is generous for levels because persistent series inflate them. Monthly comparisons "
                     "use the monthly average of the weekly index. MOVE and emerging-market spreads are not freely available back to 1994; "
                     "they will come from Bloomberg.", S_["note"])]

    end_part = [PageBreak(), Paragraph("DATA, CODE AND SOURCES", S_["eye"]),
           Paragraph(f"Every data file and the code: <b>{REPO}</b>", S_["p"]),
           Paragraph("Sources: Federal Reserve Board (federalreserve.gov); FRED, Federal Reserve Bank of St. Louis (2-year Treasury yield, CPI, "
                     "unemployment, VIX; VIX values are not redistributed); Husted, Rogers and Sun, 'Monetary Policy Uncertainty'; Baker, "
                     "Bloom and Davis, 'Measuring Economic Policy Uncertainty'; NBER recession dates.", S_["note"]),
           Paragraph("APPENDIX", S_["eye"]), Paragraph("The instructions every model received, word for word", S_["h"]),
           Preformatted(SYSTEM_PROMPT, S_["mono"], maxLineLength=118)]

    st += side_part + each_part + measures_part + [PageBreak()] + data_part + cleaning_part + prompt_part + models_part + end_part

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(muted)
        c.drawString(0.65 * inch, 0.45 * inch, "How uncertain does the Fed sound? A weekly index read by three AI models, 1994-2015 · Amaan Hafeez")
        c.drawRightString(7.85 * inch, 0.45 * inch, f"{d.page}")
        c.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.6 * inch,
                            bottomMargin=0.7 * inch, title="How uncertain does the Fed sound? A weekly index read by three AI models, 1994-2015",
                            author="Amaan Hafeez", creator="Amaan Hafeez", producer="Amaan Hafeez", subject="Fed policy uncertainty index")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
