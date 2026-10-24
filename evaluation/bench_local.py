"""Local inference latency benchmark, driving the engine directly (no HTTP).

Sweeps state length and question count on one or more devices, timing
`SystemOneEngine.evaluate` end to end: tokenisation, the torso forward(s), the readout
and the answers pulled back to Python (which forces a device sync, so the wall clock is
the real one on MPS and CUDA alike). For multi-question requests it times both the
shared-prefix path and plain batched encoding, and checks they give the same answers.

    python evaluation/bench_local.py checkpoints/hobson-2b-v19 --device mps
    python evaluation/bench_local.py checkpoints/hobson-2b-v19 --device mps --device mlx
    python evaluation/bench_local.py checkpoints/hobson-2b-v19 --device cpu \
        --lengths 256,1024 --questions 1,8 --reps 3

h2o-lightning checkpoints (docs/h2o-lightning.md) are timed the same way, on one path
(`single`: each question is one whole prompt), in the reading `--attention` gives; the
GGUF build through a llama-server with `--engine llama-server`:

    python evaluation/bench_local.py h2oai/h2o-lightning-4b --revision REV --device cuda \
        --attention one-way --image photo.png
    python evaluation/bench_local.py h2oai/h2o-lightning-4b --revision REV \
        --engine llama-server --llama-server-url http://127.0.0.1:8080

`--image` (repeatable) times requests carrying those images; the `images` column gives
their number (0 without).

Writes one CSV row per (device, length, questions, path) with median / p90 / min over
`--reps` timed runs, after `--warmup` untimed ones per device.
"""

from __future__ import annotations

import argparse
import base64
import csv
import platform
import resource
import statistics
import sys
import time
from dataclasses import replace

import torch

from strands_decider.h2o import H2OEngine
from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion, SystemOneRequest

# Deterministic, realistic-ish filler: a support-policy document, repeated and
# numbered so no two sections are byte-identical. Trimmed to an exact token count.
_SECTION = (
    "Section {n}. Refunds and payout disputes. A customer may request a refund within "
    "30 days of the charge date. Payouts that fail because of an invalid bank account "
    "are retried twice, 24 hours apart, before the account is flagged for manual "
    "review. Support agents must confirm the account holder's identity before changing "
    "payout details, and must escalate any dispute above $5,000 to the finance team. "
    "Where a policy in this section conflicts with a regional addendum, the addendum "
    "takes precedence for customers domiciled in that region. "
)
_TAIL = "\nCustomer message: Help! My payouts have been failing for 3 days."

_TEAMS = {
    "billing": "payments, invoices, payouts",
    "technical": "bugs, outages, API errors",
    "sales": "pricing and upgrades",
    "trust": "fraud, identity, account security",
}


def make_state(tok, n_tokens: int) -> str:
    tail_ids = tok(_TAIL, add_special_tokens=False)["input_ids"]
    body, n = "", 1
    while len(tok(body, add_special_tokens=False)["input_ids"]) < n_tokens:
        body += _SECTION.format(n=n)
        n += 1
    ids = tok(body, add_special_tokens=False)["input_ids"][: max(1, n_tokens - len(tail_ids))]
    return tok.decode(ids) + _TAIL


def make_questions(k: int) -> dict[str, ChoiceQuestion]:
    # Distinct instructions per question, so no two share a suffix.
    return {
        f"q{i}": ChoiceQuestion(
            type="choice",
            instructions=f"Question {i + 1}: which team should own step {i + 1} of handling this?",
            criteria=_TEAMS,
        )
        for i in range(k)
    }


def reset_peak_mem(device: str) -> None:
    """Start MLX's peak counter at this request shape; it otherwise holds the load-time peak."""
    if device == "mlx":
        import mlx.core as mx

        mx.reset_peak_memory()


def peak_mem_gib(device: str) -> float:
    if device == "mlx":
        import mlx.core as mx

        return float(mx.get_peak_memory()) / 2**30
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**30
    # ru_maxrss is bytes on macOS, KiB on Linux.
    scale = 1 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**30


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def time_request(engine, req, reps: int):
    times, resp = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.inference_mode():
            resp = engine.evaluate(req)
        times.append((time.perf_counter() - t0) * 1000)
    return times, resp


def _dtype_name(engine, device: str) -> str:
    """The dtype the torso runs in, as the CSV reports it."""
    if isinstance(engine, H2OEngine):
        lm = getattr(engine.backend, "lm", None)
        if lm is not None:
            return str(next(lm.parameters()).dtype).removeprefix("torch.")
        return {"mlx": "mlx-checkpoint", "llama-server": "gguf"}.get(engine.cfg.device, "?")
    dtype = (
        engine.model.config.torch_dtype
        if device == "mlx"
        else next(engine.model.torso.parameters()).dtype
    )
    return str(dtype).removeprefix("torch.")


def _images(paths: list[str]) -> list[str]:
    out = []
    for path in paths:
        with open(path, "rb") as fh:
            out.append(base64.b64encode(fh.read()).decode("ascii"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument(
        "--device", action="append", help="repeatable; default mps if available else cpu"
    )
    ap.add_argument("--lengths", default="256,1024,2048,4000", help="state lengths in tokens")
    ap.add_argument("--questions", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--dtype",
        choices=["checkpoint", "float32", "bfloat16", "float16"],
        default="checkpoint",
        help="cast the torso after loading (on CPU the engine already uses fp32); "
        "strands-decider checkpoints only",
    )
    ap.add_argument(
        "--attention",
        choices=["two-way", "one-way"],
        default=None,
        help="h2o-lightning checkpoints: the reading (default: the build's own)",
    )
    ap.add_argument("--revision", default=None, help="Hub revision of the checkpoint")
    ap.add_argument(
        "--image",
        action="append",
        default=[],
        help="image file sent with every request; repeatable (loads the vision tower)",
    )
    ap.add_argument("--engine", choices=["local", "llama-server"], default="local")
    ap.add_argument("--llama-server-url", default=None)
    ap.add_argument("--out", default="bench_local.csv")
    args = ap.parse_args()

    devices = args.device or ["mps" if torch.backends.mps.is_available() else "cpu"]
    if args.engine == "llama-server":
        devices = ["llama-server"]
    lengths = [int(x) for x in args.lengths.split(",")]
    qcounts = [int(x) for x in args.questions.split(",")]
    images = _images(args.image)

    rows = []
    for device in devices:
        t0 = time.perf_counter()
        engine = load_engine(
            args.checkpoint,
            device="cpu" if device == "llama-server" else device,
            attention=args.attention,
            revision=args.revision,
            vision=bool(images),
            engine=args.engine,
            llama_server_url=args.llama_server_url,
        )
        h2o = isinstance(engine, H2OEngine)
        if args.dtype != "checkpoint" and (device == "mlx" or h2o):
            raise SystemExit(
                "--dtype applies to strands-decider checkpoints on torch devices; the MLX torso "
                "and h2o-lightning checkpoints run in the checkpoint's dtype"
            )
        if args.dtype != "checkpoint":
            # load_engine builds under inference_mode, so the cast must happen there too.
            with torch.inference_mode():
                engine.model.torso.to(getattr(torch, args.dtype))
        load_s = time.perf_counter() - t0
        tok = engine.tok
        dtype = _dtype_name(engine, device)
        reading = f", {engine.attention}" if h2o else ""
        print(
            f"\n== {device}: loaded in {load_s:.1f}s, torch {torch.__version__}, "
            f"threads {torch.get_num_threads()}, torso {dtype}{reading}",
            flush=True,
        )

        warm = SystemOneRequest(
            state=make_state(tok, 256), questions=make_questions(2), images=images
        )
        for _ in range(args.warmup):
            with torch.inference_mode():
                engine.evaluate(warm)

        base_cfg = engine.cfg
        for n_tok in lengths:
            state = make_state(tok, n_tok)
            for k in qcounts:
                req = SystemOneRequest(state=state, questions=make_questions(k), images=images)
                # One question always takes the batched path (infer.py skips the prefix
                # cache for it), so there is only one path to time. An h2o-lightning engine
                # reads every question as one whole prompt: one path as well.
                paths = ["prefix", "batched"] if k > 1 and not h2o else ["single"]
                answers = {}
                for path in paths:
                    if not h2o:
                        engine.cfg = replace(base_cfg, use_prefix_cache=(path == "prefix"))
                    reset_peak_mem(device)
                    times, resp = time_request(engine, req, args.reps)
                    if path == "prefix" and not engine.cfg.use_prefix_cache:
                        # The engine fell back to batched encoding: record it, don't
                        # report batched timings under the prefix label.
                        print(
                            f"   !! prefix cache disabled itself on {device}; row marked",
                            flush=True,
                        )
                        path = "prefix-FELLBACK"
                    answers[path] = {n: a.choice for n, a in resp.answers.items()}
                    row = {
                        "device": device,
                        "dtype": dtype,
                        "state_tokens": n_tok,
                        "questions": k,
                        "path": path,
                        "input_tokens": resp.usage.input_tokens,
                        "median_ms": round(statistics.median(times), 1),
                        "p90_ms": round(pct(times, 0.9), 1),
                        "min_ms": round(min(times), 1),
                        "reps": args.reps,
                        "peak_mem_gib": round(peak_mem_gib(device), 2),
                        "images": len(images),
                    }
                    rows.append(row)
                    print(
                        f"   {n_tok:>5} tok  {k:>2} q  {path:<15} median {row['median_ms']:>9.1f} ms"
                        f"  p90 {row['p90_ms']:>9.1f}  in-tok {row['input_tokens']:>6}",
                        flush=True,
                    )
                if len(answers) == 2:
                    a, b = answers.values()
                    same = sum(a[n] == b[n] for n in a)
                    rows[-1]["same_answer"] = rows[-2]["same_answer"] = f"{same}/{k}"
                    if same != k:
                        print(f"   note: prefix vs batched agree on {same}/{k} answers", flush=True)
        engine.cfg = base_cfg
        del engine
        if device == "mps":
            torch.mps.empty_cache()

    fields = [
        "device",
        "dtype",
        "state_tokens",
        "questions",
        "path",
        "input_tokens",
        "median_ms",
        "p90_ms",
        "min_ms",
        "reps",
        "peak_mem_gib",
        "same_answer",
        "images",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
