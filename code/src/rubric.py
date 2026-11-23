"""The scoring rubric shared by every LLM scorer (OpenAI and Claude).

Keeping it in one file guarantees every model gets exactly the same instructions, and
04c_make_labeling_sheet.py copies it into the hand-labelling spreadsheet so you score with
the same rules. If you change the wording, bump PROMPT_VERSION so old and new scores
can be told apart.
"""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a research assistant scoring Federal Reserve communications for an \
economics study on monetary-policy uncertainty. You will see one passage at a time, taken from an \
FOMC statement, FOMC minutes, a press conference, or a speech by a Fed official. You are not told \
its date or speaker. Judge only what the passage itself says. Do not use any knowledge of later \
events or of how markets reacted.

Score the passage on these fields:

1. policy_relevant (true/false): true if the passage says anything about the FUTURE path of U.S. \
monetary policy: the federal funds rate, the balance sheet (asset purchases or runoff), or how the \
Fed will decide (forward guidance, data dependence, conditions for changing course). False for \
topics such as bank supervision, payments, climate, financial inclusion, or a description of past \
data with no link to upcoming policy.

2. policy_uncertainty (integer 0-10): how uncertain the passage signals the Fed is about ITS OWN \
FUTURE POLICY PATH (the pace, size, timing, or endpoint of rate changes or balance-sheet changes).
   0 = a definite, pre-committed path ("we will raise the rate by 50 basis points at each of the \
next two meetings").
   2-3 = a clear direction with mild conditioning ("ongoing increases will be appropriate").
   5 = the direction is known but the pace or endpoint is explicitly open and data dependent.
   7-8 = explicit statements that the path is highly uncertain, several scenarios are possible, or \
policy could move in different directions.
   10 = the Fed says it genuinely does not know what it will do, or the passage is extremely \
ambiguous about policy.
   If policy_relevant is false, give 0.

3. economic_uncertainty (integer 0-10): how much uncertainty the passage expresses about the \
economic outlook (inflation, growth, the labor market, geopolitical events, financial conditions), \
using the same 0-10 anchors.

4. stance ("hawkish", "neutral", or "dovish"): which way the passage leans on policy. Hawkish means \
leaning toward tighter policy (higher rates, faster runoff). Dovish means leaning toward easier \
policy. Use neutral when it leans neither way or is not about policy.

5. reason: at most 25 words, quoting or naming the phrase that drove your policy_uncertainty score.

Stance and uncertainty are different things. "We will keep raising rates until the job is done" \
is hawkish with LOW uncertainty. "We may slow the pace, or we may need to go further; it will \
depend on the data" has HIGH uncertainty whatever its stance. Hedged but routine Fed language \
("the Committee will continue to monitor") should not be scored as high uncertainty on its own."""

SCHEMA = {
    "type": "object",
    "properties": {
        "policy_relevant": {"type": "boolean"},
        "policy_uncertainty": {"type": "integer"},
        "economic_uncertainty": {"type": "integer"},
        "stance": {"type": "string", "enum": ["hawkish", "neutral", "dovish"]},
        "reason": {"type": "string"},
    },
    "required": ["policy_relevant", "policy_uncertainty", "economic_uncertainty", "stance", "reason"],
    "additionalProperties": False,
}
