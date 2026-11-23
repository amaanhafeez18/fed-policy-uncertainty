"""Step 4b (Google version): score every passage with a Gemini model, as a check from a second company.

Same setup as 04b_score_openai.py: each passage is sent ALONE, with no date and no speaker, with the
same rubric and JSON format from src/rubric.py, so the models differ only in who built them.

Usage (needs a key from aistudio.google.com/apikey in GEMINI_API_KEY):
  python src/04b_score_gemini.py --dry-run            # show the request and cost estimate, spend nothing
  python src/04b_score_gemini.py --limit 20           # cheap trial run on 20 passages
  python src/04b_score_gemini.py --years 2022-2024    # only passages released in these years
  python src/04b_score_gemini.py                      # all passages
  python src/04b_score_gemini.py --workers 1          # slower, for a free-tier key with low limits

Thinking is left at the model's default unless --thinking is given. Google's docs don't list the allowed
levels for gemini-3.1-flash-lite, and thinking tokens are billed as output.

Results are appended to data/processed/scores_llm_<model>.jsonl as they arrive, so the script can be
stopped and restarted without paying twice. It stops by itself, keeping its progress, if a quota runs out
(daily quotas reset at midnight Pacific time), if the billing balance is $0 (error 402), or if Google's
servers stay overloaded (error 503); rerun the same command once that has cleared.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google import genai
from google.genai import errors, types

from config import PROCESSED
from rubric import PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT

# Standard prices, US$ per 1M tokens (input, output), from aistudio.google.com/docs/pricing,
# checked 2026-11-04. Used only to print a cost estimate.
PRICES = {"gemini-3.1-flash-lite": (0.25, 1.50)}


def request_params(model: str, thinking: str | None, text: str) -> dict:
    config = dict(system_instruction=SYSTEM_PROMPT, response_mime_type="application/json",
                  response_json_schema=SCHEMA, max_output_tokens=4000)
    if thinking:
        config["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    return dict(model=model, contents=f"<passage>\n{text}\n</passage>", config=types.GenerateContentConfig(**config))


def score_one(client: genai.Client, model: str, thinking: str | None, text: str) -> dict:
    response = client.models.generate_content(**request_params(model, thinking, text))

    meta = response.usage_metadata
    # Thinking tokens are billed as output, so they count towards the output total.
    usage = {"input_tokens": (meta.prompt_token_count or 0) if meta else 0,
             "output_tokens": ((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) if meta else 0}
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return {"error": f"blocked_{response.prompt_feedback.block_reason.value}", **usage}
    if not response.candidates:
        return {"error": "no_candidates", **usage}
    finish = response.candidates[0].finish_reason
    if finish != types.FinishReason.STOP:
        return {"error": f"finish_{finish.value if finish else 'unknown'}", **usage}
    try:
        result = json.loads(response.text)
        for k in ("policy_uncertainty", "economic_uncertainty"):
            result[k] = min(10, max(0, int(result[k])))
        if result["stance"] not in ("hawkish", "neutral", "dovish") or not isinstance(result["policy_relevant"], bool):
            raise ValueError
        result["reason"] = str(result["reason"])
    except (TypeError, ValueError, KeyError):
        return {"error": "bad_output", **usage}
    keep = ("policy_relevant", "policy_uncertainty", "economic_uncertainty", "stance", "reason")
    return {**{k: result[k] for k in keep}, "served_by": response.model_version or model, **usage}


def estimate(passages: pd.DataFrame, model: str):
    """Rough cost estimate: ~4 characters per token; output = short JSON + some thinking."""
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
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--thinking", choices=["minimal", "low", "medium", "high"],
                    help="thinking level; leave unset to use the model's default")
    ap.add_argument("--years", help="score only passages released in this range, e.g. 2022-2024")
    ap.add_argument("--limit", type=int, help="score only the first N unscored passages (trial run)")
    ap.add_argument("--workers", type=int, default=3)
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
    print(f"{len(done)} already scored; {len(todo)} to score with {args.model} "
          f"(thinking={args.thinking or 'model default'})")
    estimate(todo, args.model)

    if args.dry_run:
        example = request_params(args.model, args.thinking, todo.iloc[0].text)
        print("\nExample request (instructions shortened):")
        print(json.dumps({"model": example["model"], "contents": example["contents"],
                          "system_instruction": SYSTEM_PROMPT[:300] + " [...]",
                          "response_mime_type": "application/json", "response_json_schema": SCHEMA}, indent=2))
        return

    # The library retries rate limits and server errors by itself before reporting a failure.
    retry = types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=60.0, jitter=1.0, http_status_codes=[429, 500, 502, 503, 504])
    client = genai.Client(http_options=types.HttpOptions(retry_options=retry))
    lock = threading.Lock()
    tokens_in = tokens_out = errs = overloaded = 0
    stop_reason = None
    every = 50 if len(todo) > 100 else 5

    with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(score_one, client, args.model, args.thinking, r.text): r.passage_id
                   for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            pid = futures[fut]
            try:
                res = fut.result()
            except errors.APIError as e:
                if e.code == 402:
                    stop_reason = ("Google says the billing balance is $0 (error 402). Add credits in AI Studio or use a key from a "
                                   "free-tier project, then rerun this command; finished passages are saved.")
                elif e.code == 429:
                    stop_reason = "Google's rate limit or quota was reached. Rerun this command later; finished passages are saved."
                elif e.code == 503:
                    overloaded += 1
                    if overloaded >= 10:
                        stop_reason = "Google's servers stayed overloaded (error 503). Try again later; finished passages are saved."
                elif e.code in (400, 401, 403) and "API key" in str(e):
                    stop_reason = "Google rejected the API key. Check GEMINI_API_KEY and try again."
                elif e.code in (400, 404):
                    stop_reason = f"Google rejected the request ({e.code}): {e.message}"
                if stop_reason:
                    pool.shutdown(cancel_futures=True)
                    break
                res = {"error": f"api_{e.code}", "message": str(e.message)[:200]}
            except OSError:
                res = {"error": "connection"}
            if "error" not in res:
                overloaded = 0
            res = {"passage_id": pid, "model": args.model, "prompt_version": PROMPT_VERSION, **res}
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                tokens_in += res.get("input_tokens", 0)
                tokens_out += res.get("output_tokens", 0)
                errs += "error" in res
            if i % every == 0 or i == len(futures):
                cost = ""
                if args.model in PRICES:
                    p_in, p_out = PRICES[args.model]
                    cost = f", about ${tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out:.2f} spent"
                print(f"  {i}/{len(futures)} done, {errs} errors, "
                      f"{tokens_in:,} input / {tokens_out:,} output tokens{cost}")

    if stop_reason:
        print(stop_reason)

    # Collapse the log into a clean table (latest successful score per passage).
    rows = [json.loads(line) for line in out_path.read_text().splitlines()] if out_path.exists() else []
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("No passages scored successfully yet; check the errors in", out_path)
        return
    df = pd.DataFrame(ok).drop_duplicates("passage_id", keep="last")
    df.to_csv(PROCESSED / f"scores_llm_{args.model}.csv", index=False)
    print(f"Saved {len(df)} scored passages to scores_llm_{args.model}.csv")


if __name__ == "__main__":
    main()
