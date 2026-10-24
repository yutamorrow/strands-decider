"""Run the Python examples of docs/h2o-lightning.md exactly as they are written.

Each example is a ```python block right after a `<!-- example: NAME -->` line. They are run as
written, one after another, each in a fresh namespace, so the docs and what is checked cannot
drift apart (run_checks.sh calls this on every machine it checks).

    python evaluation/h2o_lightning/docs_examples.py --list
    python evaluation/h2o_lightning/docs_examples.py --only text,image

The exit code is 1 when an example fails or a name given to --only is not in the docs.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "h2o-lightning.md"
NAMES = ("text", "image", "light-mlx", "light-llama")
_EXAMPLE = re.compile(
    r"^<!-- example: (?P<name>[\w-]+) -->[ \t]*\n```python[ \t]*\n(?P<code>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def extract(markdown: str) -> dict[str, str]:
    """{name: code} of every marked example, in the order the docs give them."""
    out: dict[str, str] = {}
    for m in _EXAMPLE.finditer(markdown):
        if m["name"] in out:
            raise ValueError(f"example {m['name']!r} appears twice")
        out[m["name"]] = m["code"]
    return out


def run(name: str, code: str) -> bool:
    started = time.perf_counter()
    try:
        exec(compile(code, f"<docs example {name}>", "exec"), {"__name__": "__main__"})
    except Exception:
        traceback.print_exc()
        print(f"FAIL {name} ({time.perf_counter() - started:.1f} s)", flush=True)
        return False
    print(f"PASS {name} ({time.perf_counter() - started:.1f} s)", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--docs", default=str(DOCS), help="the markdown file (default: the repo's)")
    ap.add_argument("--only", default=None, help=f"comma-separated names among {list(NAMES)}")
    ap.add_argument("--list", action="store_true", help="print the examples' names and stop")
    args = ap.parse_args(argv)
    examples = extract(Path(args.docs).read_text(encoding="utf-8"))
    if args.list:
        print("\n".join(examples))
        return 0
    names = args.only.split(",") if args.only else list(examples)
    missing = [n for n in names if n not in examples]
    if missing:
        print(f"no example {missing} in {args.docs}", file=sys.stderr)
        return 1
    results = [run(name, examples[name]) for name in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
