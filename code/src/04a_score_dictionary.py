"""Step 4a: the "old way" baseline. Count uncertainty words in each passage.

This is how newspaper-based indices work (e.g. Baker, Bloom & Davis), and it is the
benchmark the LLM measure has to beat. If the LLM index just reproduced word counts,
it would add nothing.

Word list: a subset of the Loughran & McDonald (2011) "uncertainty" category, the standard
finance-text dictionary. Policy relevance is a simple keyword screen, so the dictionary
index uses the same idea as the LLM's "is this about the future path of policy?" question.

Output: data/processed/scores_dictionary.csv
"""
import re

import pandas as pd

from config import PROCESSED

UNCERTAINTY_WORDS = re.compile(
    r"\b(?:uncertain\w*|unclear|unknown\w*|unpredictab\w*|unforeseen|ambigu\w*|risk\w*|"
    r"may|might|could|possibl\w*|perhaps|depend\w*|conditional|contingen\w*|variab\w*|"
    r"volatil\w*|fluctuat\w*|approximat\w*|tentativ\w*|doubt\w*|speculat\w*|assum\w*|"
    r"believe\w*|probabl\w*|sometime\w*|unsettled|unresolved|unexpected\w*|sudden\w*)\b",
    re.I,
)
POLICY_WORDS = re.compile(
    r"\b(?:federal funds|policy rate|interest rates?|target range|rate (?:increase|hike|cut)s?|"
    r"tighten\w*|accommodat\w*|restrictive|balance sheet|runoff|monetary policy|"
    r"forward guidance|pace of (?:increase|tightening)|terminal rate)\b",
    re.I,
)


def main():
    p = pd.read_csv(PROCESSED / "passages.csv")
    words = p["text"].str.split().str.len()
    p["dict_unc_count"] = p["text"].str.count(UNCERTAINTY_WORDS)
    p["dict_uncertainty"] = 100 * p["dict_unc_count"] / words  # uncertainty words per 100 words
    p["dict_policy_relevant"] = p["text"].str.contains(POLICY_WORDS)
    out = p[["passage_id", "dict_unc_count", "dict_uncertainty", "dict_policy_relevant"]]
    out.to_csv(PROCESSED / "scores_dictionary.csv", index=False)
    print(f"Scored {len(out)} passages. Policy-relevant by keyword: {out.dict_policy_relevant.mean():.0%}")
    print(p.groupby("doc_type")[["dict_uncertainty", "dict_policy_relevant"]].mean().round(2))


if __name__ == "__main__":
    main()
