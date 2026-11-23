"""Step 4b: have an LLM score every passage for monetary-policy uncertainty (OpenAI version).

Each passage is sent ALONE, with no date and no speaker, and the model must answer in a
fixed JSON format (structured outputs, strict mode), so every answer can be parsed and audited.
The instructions (rubric) and the JSON format live in src/rubric.py, shared with every scorer.

The model returns, per passage:
  policy_relevant       does it talk about the FUTURE path of U.S. monetary policy?
  policy_uncertainty    0-10, how unsure the Fed sounds about its own future policy path
  economic_uncertainty  0-10, how unsure it sounds about the economy (inflation, growth, ...)
  stance                hawkish / neutral / dovish (kept separate so "aggressive" isn't
                        mistaken for "uncertain")
  reason                one short sentence quoting the key phrase, for auditing

Usage (needs an OpenAI API key in OPENAI_API_KEY):
  python src/04b_score_openai.py --dry-run          # show the request and cost estimate, spend nothing
  python src/04b_score_openai.py --limit 20         # cheap trial run on 20 passages
  python src/04b_score_openai.py                    # all passages
  python src/04b_score_openai.py --years 2022-2024  # only passages released in these years
  python src/04b_score_openai.py --model gpt-6.1-terra   # a second model, for robustness

Results are appended to data/processed/scores_llm_<model>.jsonl as they arrive, so the
script can be stopped and restarted without paying twice for the same passage.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd

from config import PROCESSED
from rubric import PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT

# Standard prices, US$ per 1M tokens (input, output), from developers.openai.com/api/docs/pricing,
# checked 2026-10-28. Used only to print a cost estimate.
PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}


def request_params(model: str, effort: str, text: str) -> dict:
    return dict(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=f"<passage>\n{text}\n</passage>",
        reasoning={"effort": effort},
        text={"format": {"type": "json_schema", "name": "fed_uncertainty_score",
                         "schema": SCHEMA, "strict": True}},
        max_output_tokens=4000,
    )


def score_one(client: openai.OpenAI, model: str, effort: str, text: str) -> dict:
    response = client.responses.create(**request_params(model, effort, text))

    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.status == "incomplete":
        return {"error": f"incomplete_{response.incomplete_details.reason}", "served_by": response.model, **usage}
    for item in response.output:
        if item.type == "message" and any(c.type == "refusal" for c in item.content):
            return {"error": "refusal", "served_by": response.model, **usage}
    result = json.loads(response.output_text)
    for k in ("policy_uncertainty", "economic_uncertainty"):
        result[k] = min(10, max(0, int(result[k])))
    return {**result, "served_by": response.model, **usage}


def estimate(passages: pd.DataFrame, model: str):
    """Rough cost estimate: ~4 characters per token; output = short JSON + some reasoning."""
    tokens_in = (len(SYSTEM_PROMPT) * len(passages) + passages["text"].str.len().sum()) / 4
    out_lo, out_hi = 100 * len(passages), 500 * len(passages)
    if model in PRICES:
        p_in, p_out = PRICES[model]
        lo = tokens_in / 1e6 * p_in + out_lo / 1e6 * p_out
        hi = tokens_in / 1e6 * p_in + out_hi / 1e6 * p_out
        print(f"Estimated cost for {len(passages)} passages on {model}: ${lo:.2f} - ${hi:.2f}")
    print(f"(about {tokens_in / 1e6:.2f}M input tokens, {out_lo / 1e6:.2f}-{out_hi / 1e6:.2f}M output tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="low", choices=["none", "low", "medium", "high"],
                    help="how much the model reasons before answering; low is usually enough here")
    ap.add_argument("--years", help="score only passages released in this range, e.g. 2022-2024")
    ap.add_argument("--limit", type=int, help="score only the first N unscored passages (trial run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="print one request and a cost estimate; no API calls")
    args = ap.parse_args()

    out_path = PROCESSED / f"scores_llm_{args.model}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["passage_id"] for line in out_path.read_text().splitlines()
                if '"error"' not in line}

    passages = pd.read_csv(PROCESSED / "passages.csv")
    todo = passages[~passages["passage_id"].isin(done)]
    if args.years:
        first, last = (int(y) for y in args.years.split("-"))
        todo = todo[todo["date"].str[:4].astype(int).between(first, last)]
    # Shuffle (fixed seed) so a --limit trial covers all document types, not just January.
    todo = todo.sample(frac=1, random_state=42)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already scored; {len(todo)} to score with {args.model} (effort={args.effort})")
    estimate(todo, args.model)

    if args.dry_run:
        example = request_params(args.model, args.effort, todo.iloc[0].text)
        example["instructions"] = example["instructions"][:300] + " [...]"
        print("\nExample request (instructions shortened):")
        print(json.dumps(example, indent=2))
        return

    client = openai.OpenAI(max_retries=6)
    lock = threading.Lock()
    tokens_in = tokens_out = errors = 0

    with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(score_one, client, args.model, args.effort, r.text): r.passage_id
                   for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            pid = futures[fut]
            try:
                res = fut.result()
            except openai.AuthenticationError:
                pool.shutdown(cancel_futures=True)
                raise SystemExit("OpenAI rejected the API key. Check OPENAI_API_KEY and try again.")
            except openai.APIStatusError as e:
                res = {"error": f"api_{e.status_code}"}
            except openai.APIConnectionError:
                res = {"error": "connection"}
            res = {"passage_id": pid, "model": args.model, "prompt_version": PROMPT_VERSION, **res}
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                tokens_in += res.get("input_tokens", 0)
                tokens_out += res.get("output_tokens", 0)
                errors += "error" in res
            if i % 50 == 0 or i == len(futures):
                cost = ""
                if args.model in PRICES:
                    p_in, p_out = PRICES[args.model]
                    cost = f", about ${tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out:.2f} spent"
                print(f"  {i}/{len(futures)} done, {errors} errors, "
                      f"{tokens_in:,} input / {tokens_out:,} output tokens{cost}")

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
