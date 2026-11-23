"""Step 4b (Claude version): have an LLM score every passage for monetary-policy uncertainty.

Each passage is sent ALONE, with no date and no speaker, and the model must answer in a
fixed JSON format (structured outputs), so every answer can be parsed and audited.

The model returns, per passage:
  policy_relevant       does it talk about the FUTURE path of U.S. monetary policy?
  policy_uncertainty    0-10, how unsure the Fed sounds about its own future policy path
  economic_uncertainty  0-10, how unsure it sounds about the economy (inflation, growth, ...)
  stance                hawkish / neutral / dovish (kept separate so "aggressive" isn't
                        mistaken for "uncertain")
  reason                one short sentence quoting the key phrase, for auditing

Alternative to 04b_score_openai.py, for anyone with Anthropic API credits.
Usage (needs an Anthropic API key in ANTHROPIC_API_KEY):
  python src/04b_score_claude.py --limit 20          # cheap trial run on 20 passages
  python src/04b_score_claude.py                     # all passages with the main model
  python src/04b_score_claude.py --model claude-haiku-5-5   # second model, for robustness

Results are appended to data/processed/scores_llm_<model>.jsonl as they arrive, so the
script can be stopped and restarted without paying twice for the same passage.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

from config import PROCESSED
from rubric import PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT

# Server-side fallback (re-runs a declined request on another model) exists for Opus and
# Sonnet; Haiku has none, so the flag is only sent where it is supported.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


def score_one(client: anthropic.Anthropic, model: str, effort: str, text: str) -> dict:
    kwargs = dict(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"<passage>\n{text}\n</passage>"}],
    )
    if model in FALLBACK_MODELS:
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    response = client.beta.messages.create(**kwargs)

    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return {"error": "refusal", "served_by": response.model, **usage}
    if response.stop_reason == "max_tokens":
        return {"error": "max_tokens", "served_by": response.model, **usage}
    text_out = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text_out)
    for k in ("policy_uncertainty", "economic_uncertainty"):
        result[k] = min(10, max(0, int(result[k])))
    return {**result, "served_by": response.model, **usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="how hard the model thinks; low is usually enough for short classification")
    ap.add_argument("--limit", type=int, help="score only the first N unscored passages (trial run)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out_path = PROCESSED / f"scores_llm_{args.model}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["passage_id"] for line in out_path.read_text().splitlines()
                if '"error"' not in line}

    passages = pd.read_csv(PROCESSED / "passages.csv")
    todo = passages[~passages["passage_id"].isin(done)]
    # Shuffle (fixed seed) so a --limit trial covers all document types, not just January.
    todo = todo.sample(frac=1, random_state=42)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already scored; scoring {len(todo)} passages with {args.model} (effort={args.effort})")

    client = anthropic.Anthropic(max_retries=6)
    lock = threading.Lock()
    tokens_in = tokens_out = errors = 0

    with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(score_one, client, args.model, args.effort, r.text): r.passage_id
                   for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            pid = futures[fut]
            try:
                res = fut.result()
            except anthropic.APIStatusError as e:
                res = {"error": f"api_{e.status_code}"}
            except anthropic.APIConnectionError:
                res = {"error": "connection"}
            res = {"passage_id": pid, "model": args.model, "prompt_version": PROMPT_VERSION, **res}
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                tokens_in += res.get("input_tokens", 0)
                tokens_out += res.get("output_tokens", 0)
                errors += "error" in res
            if i % 50 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} done, {errors} errors, "
                      f"{tokens_in:,} input / {tokens_out:,} output tokens so far")

    # Collapse the log into a clean table (latest successful score per passage).
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("No passages scored successfully yet; check the errors in", out_path)
        return
    df = pd.DataFrame(ok).drop_duplicates("passage_id", keep="last")
    df.to_csv(PROCESSED / f"scores_llm_{args.model}.csv", index=False)
    print(f"Saved {len(df)} scored passages to scores_llm_{args.model}.csv")


if __name__ == "__main__":
    main()
