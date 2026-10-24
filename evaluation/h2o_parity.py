"""Do two ways of serving h2o-lightning give the same answers? Fixed requests, one table row.

The request set is docs/h2o-lightning.md's 12 text requests (4 yes/no, 4 choice, 4 score) and,
unless `--text-only`, 4 requests over images drawn here with Pillow (shapes and colours; no
image file is shipped). Answers come from an engine in this process (`load_engine`'s arguments)
or from any server's `POST /v1/systemone` (`--url`, e.g. h2o's own serving or `strands-decider
serve`), and are written to `--out` as JSON.

    python evaluation/h2o_parity.py h2oai/h2o-lightning-4b --revision REV --device cuda \
        --attention one-way --out hf-cuda-one-way.json
    python evaluation/h2o_parity.py --url http://127.0.0.1:8741 --text-only --out shim.json
    python evaluation/h2o_parity.py h2oai/h2o-lightning-4b --revision REV \
        --engine llama-server --out gguf-q8.json \
        --compare hf-cuda-one-way.json --max-dp 0.05 --label C2

`--compare REF.json` compares the answers with REF's over the requests both have and prints one
markdown row: label, answers that agree / requests compared, the largest probability difference,
PASS or FAIL. An answer agrees when the chosen option is the same (yes/no: the side of 0.5;
choice: the most likely option; score: the most likely level). The exit code is 1 on FAIL.
`--answers FILE` compares an existing answer file instead of producing one.

`--details FILE` (with `--compare`) also writes the comparison per request to FILE, as a markdown
table: the request, REF's answer and this one, whether they agree, the largest probability
difference on that request and, where the answers differ, the three most likely options of each
with their probabilities. It changes nothing in the row or the exit code.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request
from typing import Any

TEXT_REQUESTS: dict[str, dict[str, Any]] = {
    "noul-outage": {
        "state": "Checkout returns HTTP 500 for every customer since 09:12.",
        "questions": {"q": {"type": "noul", "instructions": "Is this a full outage?"}},
    },
    "noul-parcel": {
        "state": "The parcel left the depot on Monday, was scanned in Lyon on Tuesday and has "
        "not moved since.",
        "questions": {"q": {"type": "noul", "instructions": "Should the parcel be declared lost?"}},
    },
    "noul-invoice": {
        "state": {
            "from": "dana@example.com",
            "subject": "Invoice 4471",
            "body": "Please find the overdue invoice attached.",
        },
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "Does this email ask the reader to pay something?",
            }
        },
    },
    "noul-code": {
        "state": "def add(a, b):\n    return a - b",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "Does the function do what its name says?",
                "criteria": {"true": "it adds its arguments", "false": "it does something else"},
            }
        },
    },
    "choice-team": {
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "payments and payouts",
                    "sales": "new contracts",
                    "retail": "store orders",
                },
            }
        },
    },
    "choice-language": {
        "state": "sihamba ngokushesha",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "What language is this phrase in?",
                "criteria": {"english": "English", "zulu": "Zulu", "dutch": "Dutch"},
            }
        },
    },
    "choice-tool": {
        "state": {"tool": "get_weather", "arguments": {"city": ""}},
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "What should the agent do with this tool call?",
                "criteria": {
                    "run": "run it as is",
                    "ask": "ask the user for the missing city",
                    "drop": "drop the call",
                },
            }
        },
    },
    "choice-complaint": {
        "state": "Battery lasts two days, screen is dim outdoors, support replaced it once "
        "without fuss.",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "What is the main complaint?",
                "criteria": {
                    "battery": "Battery",
                    "screen": "Screen",
                    "support": "Support",
                    "none": "No complaint",
                },
            }
        },
    },
    "score-positive": {
        "state": "this is the best doc I've ever read",
        "questions": {
            "q": {
                "type": "score",
                "instructions": "How positive is this?",
                "criteria": ["negative", "neutral", "positive"],
            }
        },
    },
    "score-frustrated": {
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {
            "q": {
                "type": "score",
                "instructions": "How frustrated is the writer?",
                "criteria": ["calm", "frustrated", "depressed"],
            }
        },
    },
    "score-urgent": {
        "state": "The server is at 92% disk and rising 1% an hour.",
        "questions": {
            "q": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "today", "within the hour", "now"],
            }
        },
    },
    "score-supported": {
        "state": "The answer cites two sources; one of them does not mention the claim.",
        "questions": {
            "q": {
                "type": "score",
                "instructions": "How well supported is the answer?",
                "criteria": ["unsupported", "partly supported", "fully supported"],
            }
        },
    },
}


def draw(shape: str, color: str, size: tuple[int, int] = (448, 336)) -> str:
    """A base64 PNG of one filled `shape` ("circle", "square" or "triangle") on white."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    w, h = size
    box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon([(w // 2, h // 4), (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_requests() -> dict[str, dict[str, Any]]:
    """Requests over drawn images, one image each: every question type."""
    shapes = {
        "circle": "a circle",
        "square": "a square",
        "triangle": "a triangle",
    }
    return {
        "image-noul-circle": {
            "state": "A picture from the shape sorter.",
            "images": [draw("circle", "red")],
            "questions": {"q": {"type": "noul", "instructions": "Is there a circle?"}},
        },
        "image-choice-shape": {
            "state": "A picture from the shape sorter.",
            "images": [draw("triangle", "blue")],
            "questions": {
                "q": {"type": "choice", "instructions": "Which shape is shown?", "criteria": shapes}
            },
        },
        "image-score-red": {
            "state": "A picture from the shape sorter.",
            "images": [draw("square", "green")],
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "How red is the shape?",
                    "criteria": ["not red", "somewhat red", "red"],
                }
            },
        },
        "image-choice-color": {
            "state": "A picture from the shape sorter.",
            "images": [draw("circle", "green")],
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "What colour is the shape?",
                    "criteria": {"red": "red", "green": "green", "blue": "blue"},
                }
            },
        },
    }


def requests(text_only: bool = False) -> dict[str, dict[str, Any]]:
    return dict(TEXT_REQUESTS) if text_only else {**TEXT_REQUESTS, **image_requests()}


def distribution(answer: dict[str, Any]) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": float(answer["noul"]), "false": 1.0 - float(answer["noul"])}
    return {k: float(v) for k, v in answer["probabilities"].items()}


def chosen(answer: dict[str, Any]) -> str:
    d = distribution(answer)
    return max(d, key=d.__getitem__)


def from_engine(engine: Any, bodies: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from strands_decider.schema import SystemOneRequest

    return {
        name: engine.evaluate(SystemOneRequest.model_validate(body)).model_dump()["answers"]["q"]
        for name, body in bodies.items()
    }


def from_url(
    url: str, bodies: dict[str, dict[str, Any]], timeout: float = 120.0
) -> dict[str, dict[str, Any]]:
    # The server is one started for the check (usually on 127.0.0.1): never go through a proxy
    # from the environment (http_proxy, https_proxy, all_proxy).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    out = {}
    for name, body in bodies.items():
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/systemone",
            json.dumps(body).encode("utf-8"),
            {"content-type": "application/json"},
        )
        with opener.open(req, timeout=timeout) as reply:
            out[name] = json.load(reply)["answers"]["q"]
    return out


def compare(
    reference: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Agreement and the largest probability difference over the requests both answered."""
    names = [n for n in reference if n in other]
    agree, worst = 0, 0.0
    for name in names:
        a, b = reference[name], other[name]
        da, db = distribution(a), distribution(b)
        if set(da) != set(db):
            raise SystemExit(f"{name}: the two answers have different options")
        agree += chosen(a) == chosen(b)
        worst = max(worst, *(abs(da[k] - db[k]) for k in da))
    return {"agree": agree, "compared": len(names), "max_abs_dp": round(worst, 4)}


def top(answer: dict[str, Any], n: int = 3) -> list[tuple[str, float]]:
    """The `n` most likely options of an answer, most likely first."""
    return sorted(distribution(answer).items(), key=lambda kv: kv[1], reverse=True)[:n]


def per_request(
    reference: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """`compare` request by request, over the same requests: both answers, whether they agree,
    the largest probability difference on that request, and the top 3 options of each."""
    rows = []
    for name in [n for n in reference if n in other]:
        a, b = reference[name], other[name]
        da, db = distribution(a), distribution(b)
        if set(da) != set(db):
            raise SystemExit(f"{name}: the two answers have different options")
        rows.append(
            {
                "request": name,
                "reference": chosen(a),
                "answers": chosen(b),
                "agree": chosen(a) == chosen(b),
                "max_abs_dp": round(max(abs(da[k] - db[k]) for k in da), 4),
                "reference_top": top(a),
                "answers_top": top(b),
            }
        )
    return rows


def _cell(text: Any) -> str:
    """A table cell: a "|" in the text would end the cell, so it becomes "/"."""
    return str(text).replace("|", "/")


def _tops(options: list[tuple[str, float]]) -> str:
    return ", ".join(f"{_cell(k)} {p:.4f}" for k, p in options)


def detail_rows(rows: list[dict[str, Any]]) -> list[str]:
    """`per_request` as a markdown table; the top 3 options only where the answers differ."""
    out = [
        "| request | reference | answers | agree | largest difference "
        "| reference top 3 | answers top 3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tops = ("-", "-") if r["agree"] else (_tops(r["reference_top"]), _tops(r["answers_top"]))
        out.append(
            f"| {_cell(r['request'])} | {_cell(r['reference'])} | {_cell(r['answers'])} "
            f"| {'yes' if r['agree'] else 'NO'} | {r['max_abs_dp']:.4f} | {tops[0]} | {tops[1]} |"
        )
    return out


def table_row(label: str, result: dict[str, Any], max_dp: float) -> tuple[str, bool]:
    ok = result["compared"] > 0 and result["agree"] == result["compared"]
    ok = ok and result["max_abs_dp"] <= max_dp
    row = (
        f"| {label} | {result['agree']}/{result['compared']} | {result['max_abs_dp']:.4f} "
        f"(limit {max_dp:g}) | {'PASS' if ok else 'FAIL'} |"
    )
    return row, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", nargs="?", help="answer with an engine on this checkpoint")
    ap.add_argument("--url", help="answer with this server's POST /v1/systemone instead")
    ap.add_argument("--answers", help="compare this existing answer file instead")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--attention", choices=["two-way", "one-way"], default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--engine", choices=["local", "llama-server"], default="local")
    ap.add_argument("--llama-server-url", default=None)
    ap.add_argument("--text-only", action="store_true", help="leave out the image requests")
    ap.add_argument("--out", help="write the answers here (JSON)")
    ap.add_argument("--compare", metavar="REF.json", help="compare the answers with REF's")
    ap.add_argument("--max-dp", type=float, default=0.02, help="largest allowed difference")
    ap.add_argument("--label", default="comparison", help="the row's first column")
    ap.add_argument("--details", metavar="FILE", help="write the comparison per request here")
    args = ap.parse_args(argv)
    sources = [x for x in (args.checkpoint, args.url, args.answers) if x]
    if len(sources) != 1:
        ap.error("give exactly one of CHECKPOINT, --url and --answers")
    if args.details and not args.compare:
        ap.error("--details needs --compare")

    bodies = requests(args.text_only)
    if args.answers:
        with open(args.answers, encoding="utf-8") as fh:
            answers = json.load(fh)["answers"]
        source = args.answers
    elif args.url:
        answers = from_url(args.url, bodies)
        source = args.url
    else:
        from strands_decider.infer import load_engine

        engine = load_engine(
            args.checkpoint,
            device=args.device,
            attention=args.attention,
            revision=args.revision,
            vision=not args.text_only,
            engine=args.engine,
            llama_server_url=args.llama_server_url,
        )
        answers = from_engine(engine, bodies)
        source = args.checkpoint
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"source": source, "answers": answers}, fh, indent=1)
    if not args.compare:
        print(f"answered {len(answers)} requests from {source}")
        return 0
    with open(args.compare, encoding="utf-8") as fh:
        reference = json.load(fh)["answers"]
    row, ok = table_row(args.label, compare(reference, answers), args.max_dp)
    if args.details:
        with open(args.details, "w", encoding="utf-8") as fh:
            fh.write("\n".join(detail_rows(per_request(reference, answers))) + "\n")
    print(row)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
