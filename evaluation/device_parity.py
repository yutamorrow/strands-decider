"""Do two devices give the same answers? One checkpoint, a fixed request set, one device at a time.

The request set covers both of the engine's paths: one question per request (the whole prompt
in one forward) and five per request (the shared-prefix path, which hands Gated DeltaNet state
across), states from about 40 to about 3,000 tokens, and every question type. Each device's
answers are compared against the first device's: the largest probability difference, and how
many chosen answers changed.

    python evaluation/device_parity.py checkpoints/hobson-2b-v19 --device mps --device mlx
    python evaluation/device_parity.py StrandsAgents/strands-decider-2B-hobson-v19 \
        --device cpu --device mlx --out parity.json
    python evaluation/device_parity.py h2oai/h2o-lightning-4b --revision REV \
        --device cpu --device mlx --attention one-way

Devices load one after another, and each engine is released before the next loads.
"""

from __future__ import annotations

import argparse
import gc
import json

from strands_decider.infer import load_engine
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

POLICY = (
    "Refunds and payout disputes. A customer may request a refund within 30 days of the "
    "charge date. Payouts that fail because of an invalid bank account are retried twice, 24 "
    "hours apart, before the account is flagged for manual review. Agents must confirm the "
    "account holder's identity before changing payout details, and must escalate any dispute "
    "above $5,000 to the finance team. "
)
TICKETS = [
    "Hi, can you tell me when my next payout is scheduled?",
    "My payouts have failed for three days and I have rent due tomorrow. This is unacceptable. "
    "I already updated my bank details twice.",
    "I was charged twice for the same order on the 3rd. I'd like one of the charges refunded, "
    "the order number is 4471.",
]
QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "charges, refunds, payouts",
            "technical": "bugs and outages",
            "account": "identity and settings",
            "sales": "new purchases",
        },
    ),
    "urgent": NoulQuestion(instructions="Does the customer convey urgency?"),
    "mood": ScoreQuestion(
        instructions="How frustrated is the customer?",
        criteria=["calm", "mildly annoyed", "frustrated", "angry"],
    ),
    "action": ChoiceQuestion(
        instructions="What should the agent do next?",
        criteria={
            "answer": "answer from policy",
            "refund": "issue a refund",
            "verify": "confirm the account holder's identity",
            "escalate": "escalate to finance",
            "ask": "ask a clarifying question",
        },
    ),
    "allowed": NoulQuestion(instructions="Does the policy allow a refund here?"),
}


def requests() -> dict[str, SystemOneRequest]:
    out = {}
    for t, ticket in enumerate(TICKETS):
        for repeats in (0, 4, 30):
            state = POLICY * repeats + ticket
            out[f"t{t}-p{repeats}-q1"] = SystemOneRequest(
                state=state, questions={"team": QUESTIONS["team"]}
            )
            out[f"t{t}-p{repeats}-q5"] = SystemOneRequest(state=state, questions=QUESTIONS)
    return out


def probabilities(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return dict(answer["probabilities"])


def chosen(answer: dict) -> str:
    p = probabilities(answer)
    return max(p, key=p.__getitem__)


def answers(
    checkpoint: str, device: str, attention: str | None = None, revision: str | None = None
) -> dict[str, dict]:
    engine = load_engine(checkpoint, device=device, attention=attention, revision=revision)
    try:
        return {name: engine.evaluate(req).model_dump() for name, req in requests().items()}
    finally:
        del engine
        gc.collect()


def compare(reference: dict[str, dict], other: dict[str, dict]) -> dict[str, float | int]:
    worst, changed, total = 0.0, 0, 0
    for name, response in reference.items():
        if other[name]["usage"] != response["usage"]:
            raise SystemExit(
                f"{name}: token counts differ: {other[name]['usage']} vs {response['usage']}"
            )
        for question, ref in response["answers"].items():
            got = other[name]["answers"][question]
            p, r = probabilities(got), probabilities(ref)
            worst = max(worst, max(abs(p[k] - r[k]) for k in r))
            changed += chosen(got) != chosen(ref)
            total += 1
    return {"max_abs_dp": round(worst, 4), "changed_answers": changed, "answers": total}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument(
        "--device", action="append", required=True, help="repeatable; the first is the reference"
    )
    ap.add_argument(
        "--out", default=None, help="write every device's answers and the comparison here"
    )
    ap.add_argument(
        "--attention",
        choices=["two-way", "one-way"],
        default=None,
        help="h2o-lightning checkpoints: the reading on every device (default: the build's own)",
    )
    ap.add_argument("--revision", default=None, help="Hub revision of the checkpoint")
    args = ap.parse_args()
    if len(args.device) < 2:
        ap.error("give at least two --device values")
    results = {
        device: answers(args.checkpoint, device, args.attention, args.revision)
        for device in args.device
    }
    reference = args.device[0]
    report = {device: compare(results[reference], results[device]) for device in args.device[1:]}
    for device, row in report.items():
        print(
            f"{device} vs {reference}: max |dp| {row['max_abs_dp']:.4f}, "
            f"changed answers {row['changed_answers']}/{row['answers']}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "reference": reference,
                    "comparison": report,
                    "answers": results,
                },
                fh,
                indent=1,
            )


if __name__ == "__main__":
    main()
