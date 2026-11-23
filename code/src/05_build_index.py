"""Step 5: turn passage scores into a weekly uncertainty index.

Aggregation, in two stages (each is a defensible choice, stated here so it can be argued):
  1. passage -> document: average over the passages that are about monetary policy.
     Each document counts as ONE piece of Fed communication, so a 9,000-word set of minutes
     doesn't outweigh a 400-word statement just because it is longer.
  2. document -> week (weeks end on Friday): average over documents released that week.
Weeks with no Fed communication (e.g. the pre-meeting "blackout" when officials don't speak)
carry the previous week's value forward: the market's latest read of the Fed stays the same
until the Fed says something new. Column `new_communication` marks which weeks are real.

Every available scorer is built the same way: the dictionary baseline and each LLM model.
Outputs: data/processed/index_documents.csv, index_weekly.csv
"""
import pandas as pd

from config import END_YEAR, FIRST_WEEK, PROCESSED, TODAY


def load_scores() -> tuple[pd.DataFrame, dict[str, tuple[str, str | None]]]:
    """Merge every scorer onto the passages. Returns the table and {index_name: (score_col, relevance_col)}."""
    p = pd.read_csv(PROCESSED / "passages.csv", parse_dates=["date"])
    p = p.merge(pd.read_csv(PROCESSED / "scores_dictionary.csv"), on="passage_id")
    specs = {"dictionary": ("dict_uncertainty", "dict_policy_relevant")}

    for path in sorted(PROCESSED.glob("scores_llm_*.csv")):
        model = path.stem.removeprefix("scores_llm_")
        tag = model.replace("claude-", "").replace("-", "_")
        s = pd.read_csv(path)[["passage_id", "policy_relevant", "policy_uncertainty",
                               "economic_uncertainty", "stance"]]
        coverage = p["passage_id"].isin(s["passage_id"]).mean()
        if coverage < 0.95:  # a partly-scored model would give a misleading index
            print(f"Skipping {model}: only {coverage:.0%} of passages scored "
                  f"(run 04b_score_openai.py --model {model} to finish it)")
            continue
        s = s.rename(columns={c: f"{tag}_{c}" for c in s.columns if c != "passage_id"})
        s[f"{tag}_hawkish"] = s[f"{tag}_stance"].map({"hawkish": 1, "neutral": 0, "dovish": -1})
        p = p.merge(s, on="passage_id", how="left")
        specs[f"llm_{tag}_policy"] = (f"{tag}_policy_uncertainty", f"{tag}_policy_relevant")
        specs[f"llm_{tag}_economic"] = (f"{tag}_economic_uncertainty", None)
        specs[f"llm_{tag}_hawkish"] = (f"{tag}_hawkish", f"{tag}_policy_relevant")
    return p, specs


def main():
    p, specs = load_scores()

    docs = p.groupby("doc_id")[["date", "doc_type", "title"]].first()
    for name, (score, relevant) in specs.items():
        sub = p[p[relevant].fillna(False).astype(bool)] if relevant else p
        docs[name] = sub.groupby("doc_id")[score].mean()
        docs[f"{name}_n"] = sub.groupby("doc_id")[score].count()
    docs.to_csv(PROCESSED / "index_documents.csv")

    names = list(specs)
    last = min(pd.Timestamp(f"{END_YEAR}-12-31"), pd.Timestamp(TODAY))
    weeks = pd.date_range(pd.Timestamp(FIRST_WEEK), last, freq="W-FRI")
    raw = docs.set_index("date")[names].resample("W-FRI").mean()
    # Carry values forward BEFORE cutting to the sample, so the first week keeps the latest value
    # released before it (the historical run collects from Nov 1993 for exactly this reason).
    weekly = raw.reindex(raw.index.union(weeks)).ffill().reindex(weeks)
    weekly["new_communication"] = raw[names[0]].reindex(weeks).notna()
    weekly["n_documents"] = docs.set_index("date")[names[0]].resample("W-FRI").count().reindex(weeks).fillna(0)
    weekly[names] = weekly[names].bfill()
    weekly.index.name = "week_ending"
    weekly.to_csv(PROCESSED / "index_weekly.csv")

    print(f"Built {len(names)} indices: {names}")
    print(f"Weeks with new Fed communication: {weekly.new_communication.sum()} of {len(weekly)}")
    print(weekly[names].describe().round(2).loc[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
