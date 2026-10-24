"""The timing and comparison tools on h2o-lightning checkpoints (evaluation/bench_local.py,
evaluation/device_parity.py, evaluation/h2o_parity.py), on the tiny checkpoints of the other
tests and a fake llama-server. Nothing is downloaded.
"""

from __future__ import annotations

import base64
import csv
import json
import re
import sys

import pytest

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

import bench_local  # noqa: E402
import device_parity  # noqa: E402
import h2o_parity  # noqa: E402
from test_cli_exit import _tiny_checkpoint  # noqa: E402
from test_h2o import make_checkpoint  # noqa: E402
from test_h2o_llama import FakeLlamaServer  # noqa: E402

DECIDER_FIELDS = [
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
]
BENCH = ["--lengths", "16", "--questions", "1,2", "--reps", "1", "--warmup", "0"]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return make_checkpoint(tmp_path_factory.mktemp("h2o-eval"))


@pytest.fixture
def fake():
    server = FakeLlamaServer()
    yield server
    server.close()


def _bench(monkeypatch, tmp_path, *args):
    out = tmp_path / "bench.csv"
    monkeypatch.setattr(sys, "argv", ["bench_local.py", *args, *BENCH, "--out", str(out)])
    assert bench_local.main() == 0
    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def test_bench_times_an_h2o_checkpoint_on_one_path(ckpt, tmp_path, monkeypatch):
    """An h2o-lightning checkpoint is timed on the `single` path, with no image (0)."""
    _, rows = _bench(monkeypatch, tmp_path, str(ckpt), "--device", "cpu")
    assert [(r["questions"], r["path"], r["images"]) for r in rows] == [
        ("1", "single", "0"),
        ("2", "single", "0"),
    ]
    assert rows[0]["dtype"] == "float32"
    monkeypatch.setattr(sys, "argv", ["bench_local.py", str(ckpt), "--dtype", "bfloat16", *BENCH])
    with pytest.raises(SystemExit, match="strands-decider checkpoints"):
        bench_local.main()


def test_bench_times_the_gguf_build_through_a_llama_server(ckpt, tmp_path, monkeypatch, fake):
    """`--engine llama-server` times the requests sent to the server."""
    _, rows = _bench(
        monkeypatch,
        tmp_path,
        str(ckpt),
        "--engine",
        "llama-server",
        "--llama-server-url",
        fake.url,
    )
    assert {(r["device"], r["dtype"], r["path"]) for r in rows} == {
        ("llama-server", "gguf", "single")
    }
    assert len(fake.bodies) == 1 + 2  # one question, then two


def test_bench_keeps_the_decider_columns_and_paths(tmp_path, monkeypatch):
    """For a strands-decider checkpoint the columns are as before plus `images`, and two
    questions are timed on the prefix and batched paths."""
    decider = _tiny_checkpoint(tmp_path)
    fields, rows = _bench(monkeypatch, tmp_path, decider, "--device", "cpu")
    assert fields == [*DECIDER_FIELDS, "images"]
    assert [(r["questions"], r["path"]) for r in rows] == [
        ("1", "single"),
        ("2", "prefix"),
        ("2", "batched"),
    ]
    assert rows[1]["same_answer"] == "2/2" and {r["images"] for r in rows} == {"0"}


def test_bench_times_image_requests_in_the_reading_asked_for(tmp_path, monkeypatch):
    """`--image` sends the image with every request (images column 1) and loads the vision
    tower; `--attention` reaches the engine."""
    if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
        pytest.skip("image input needs transformers >= 5.18")
    from test_h2o_vision import _picture, make_vision_checkpoint

    vckpt = make_vision_checkpoint(tmp_path / "vision")
    image = tmp_path / "shape.png"
    image.write_bytes(base64.b64decode(_picture("red")))
    seen = []
    original = bench_local.load_engine

    def spy(*args, **kwargs):
        engine = original(*args, **kwargs)
        seen.append((kwargs["vision"], engine.attention))
        return engine

    monkeypatch.setattr(bench_local, "load_engine", spy)
    _, rows = _bench(
        monkeypatch,
        tmp_path,
        str(vckpt),
        "--device",
        "cpu",
        "--attention",
        "one-way",
        "--image",
        str(image),
    )
    assert seen == [(True, "one-way")]
    assert {r["images"] for r in rows} == {"1"}


def test_device_parity_reads_an_h2o_checkpoint_in_the_given_reading(ckpt, monkeypatch, capsys):
    """Two runs of the same device in one-way agree exactly."""
    short = {k: v for k, v in device_parity.requests().items() if "-p0-" in k}
    for request in short.values():  # the tiny checkpoint has 4 labels
        request.questions = {
            n: q for n, q in request.questions.items() if q.type != "choice" or len(q.criteria) <= 4
        }
    monkeypatch.setattr(device_parity, "requests", lambda: short)
    loaded = []
    original = device_parity.load_engine

    def spy(*args, **kwargs):
        engine = original(*args, **kwargs)
        loaded.append(engine.attention)
        return engine

    monkeypatch.setattr(device_parity, "load_engine", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "device_parity.py",
            str(ckpt),
            "--device",
            "cpu",
            "--device",
            "cpu",
            "--attention",
            "one-way",
        ],
    )
    device_parity.main()
    assert loaded == ["one-way", "one-way"]
    out = capsys.readouterr().out
    assert "max |dp| 0.0000, changed answers 0/" in out


def test_h2o_parity_answers_and_compares(ckpt, tmp_path, capsys):
    """The 12 text requests are answered and written; an identical file passes, a changed answer
    or a difference over the limit fails, with one markdown row."""
    ref = tmp_path / "ref.json"
    assert h2o_parity.main([str(ckpt), "--device", "cpu", "--text-only", "--out", str(ref)]) == 0
    answers = json.loads(ref.read_text())["answers"]
    assert set(answers) == set(h2o_parity.TEXT_REQUESTS) and len(answers) == 12
    capsys.readouterr()
    assert h2o_parity.main(["--answers", str(ref), "--compare", str(ref), "--label", "C0"]) == 0
    assert capsys.readouterr().out.strip() == "| C0 | 12/12 | 0.0000 (limit 0.02) | PASS |"

    flipped = json.loads(ref.read_text())
    noul = flipped["answers"]["noul-outage"]
    noul["noul"] = 1 - noul["noul"] if abs(noul["noul"] - 0.5) > 0.01 else 0.9
    path = tmp_path / "flipped.json"
    path.write_text(json.dumps(flipped))
    assert h2o_parity.main(["--answers", str(path), "--compare", str(ref), "--max-dp", "1"]) == 1
    assert "11/12" in capsys.readouterr().out

    nudged = json.loads(ref.read_text())
    probs = nudged["answers"]["choice-team"]["probabilities"]
    first, second = list(probs)[:2]
    top = max(probs, key=probs.get)
    other = second if top == first else first
    probs[other] = probs[other] + 0.001  # a tiny move: the chosen option stays
    path.write_text(json.dumps(nudged))
    assert (
        h2o_parity.main(["--answers", str(path), "--compare", str(ref), "--max-dp", "0.0001"]) == 1
    )
    assert "12/12" in capsys.readouterr().out


def _answer_files(tmp_path):
    """Two answer files over four requests, two of which choose differently (one has a "|" in
    an option)."""
    ref = {
        "noul-outage": {"type": "noul", "noul": 0.9},
        "choice-tool": {
            "type": "choice",
            "probabilities": {"run": 0.30, "ask": 0.62, "drop": 0.08},
        },
        "score-urgent": {
            "type": "score",
            "probabilities": {"not urgent": 0.1, "today": 0.6, "within the hour": 0.2, "now": 0.1},
        },
        "choice-pipe": {"type": "choice", "probabilities": {"a|b": 0.6, "c": 0.4}},
    }
    other = json.loads(json.dumps(ref))
    other["noul-outage"]["noul"] = 0.88
    other["choice-tool"]["probabilities"] = {"run": 0.58, "ask": 0.40, "drop": 0.02}
    other["choice-pipe"]["probabilities"] = {"a|b": 0.3, "c": 0.7}
    ref_path, other_path = tmp_path / "ref-text.json", tmp_path / "gguf-Q4_K_M-text.json"
    ref_path.write_text(json.dumps({"answers": ref}))
    other_path.write_text(json.dumps({"answers": other}))
    return ref_path, other_path


DETAIL_ROWS = [
    "| request | reference | answers | agree | largest difference "
    "| reference top 3 | answers top 3 |",
    "|---|---|---|---|---|---|---|",
    "| noul-outage | true | true | yes | 0.0200 | - | - |",
    "| choice-tool | ask | run | NO | 0.2800 | ask 0.6200, run 0.3000, drop 0.0800 "
    "| run 0.5800, ask 0.4000, drop 0.0200 |",
    "| score-urgent | today | today | yes | 0.0000 | - | - |",
    "| choice-pipe | a/b | c | NO | 0.3000 | a/b 0.6000, c 0.4000 | c 0.7000, a/b 0.3000 |",
]


def test_the_comparison_per_request_names_the_answers_that_differ_and_their_top_options(
    tmp_path, capsys
):
    """h2o_parity.py --details writes one row per request (both answers, agree, the request's
    largest difference) and, where the answers differ, the top 3 options of each; a "|" in an
    option becomes "/". The table row and the exit code are the same as without --details.
    In run_checks.sh, C2-C4 (compare_gguf) and the record-only rows put that table in
    results.md after the C table and each request on the console as a CHECK-DIFF line; C1 and a
    row whose files are missing get none."""
    ref, other = _answer_files(tmp_path)
    base = ["--answers", str(other), "--compare", str(ref), "--max-dp", "0.1", "--label", "C3"]
    assert h2o_parity.main(base) == 1
    row = capsys.readouterr().out
    details = tmp_path / "details.md"
    assert h2o_parity.main([*base, "--details", str(details)]) == 1
    assert capsys.readouterr().out == row == "| C3 | 2/4 | 0.3000 (limit 0.1) | FAIL |\n"
    assert details.read_text().splitlines() == DETAIL_ROWS
    with pytest.raises(SystemExit):
        h2o_parity.main(["--answers", str(other), "--details", str(details)])

    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    results = tmp_path / "results.md"
    script = "\n".join(
        [
            "set -uo pipefail",
            f'OUT="{out}" PY="{sys.executable}" EVAL="{RUN_CHECKS.parents[1]}" FAILED=0',
            'GGUF_SOURCE_STOP=""',
            f'say() {{ printf "%s\\n" "$*" | tee -a "{results}"; }}',
            _shell_functions("C_ROWS=()", "bench() {"),
            f'compare_gguf C3 "Q4_K_M text" "{other}" "{ref}" 0.10',
            f'record C-gguf-Q4_K_M-vs-v1.3.1 "v1.3.1" "{other}" "{ref}"',
            f'compare_gguf C2 "Q8_0 text" "{tmp_path}/none.json" "{ref}" 0.05',
            f'compare C1 "one-way" "{other}" "{ref}" 1',
            "breakdown_section",
            'echo "failed=$FAILED"',
            'printf "%s\\n" "${C_ROWS[@]}"',
        ]
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "failed=1" in done.stdout
    assert "| C3 Q4_K_M text | 2/4 | 0.3000 (limit 0.1) | FAIL |" in done.stdout  # as before
    assert "| C-gguf-Q4_K_M-vs-v1.3.1 v1.3.1 | 2/4 | 0.3000 | record only |" in done.stdout
    kept = results.read_text()
    assert kept.startswith("## C: answers per request (C2-C4 and the record-only rows)\n")
    for heading in ("### C3 Q4_K_M text\n", "### C-gguf-Q4_K_M-vs-v1.3.1 v1.3.1\n"):
        section = kept.split(heading, 1)[1].split("###", 1)[0]
        assert "Answers `gguf-Q4_K_M-text.json` against the reference `ref-text.json`." in section
        assert [ln for ln in section.splitlines() if ln.startswith("|")] == DETAIL_ROWS
    assert "### C2" not in kept and "### C1" not in kept
    diffs = [ln for ln in done.stderr.splitlines() if ln.startswith("CHECK-DIFF ")]
    assert len(diffs) == 8  # 4 requests, C3 and the record-only row
    assert diffs[1] == f"CHECK-DIFF C3 gguf-Q4_K_M-text {DETAIL_ROWS[3]}"
    assert diffs[4] == f"CHECK-DIFF C-gguf-Q4_K_M-vs-v1.3.1 gguf-Q4_K_M-text {DETAIL_ROWS[2]}"


import http.server  # noqa: E402
import threading  # noqa: E402

PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy")


def test_h2o_parity_asks_a_local_server_without_environment_proxies(monkeypatch):
    """http_proxy/https_proxy/all_proxy (either case) pointing at an unreachable proxy change
    nothing: h2o_parity --url still reaches the server on 127.0.0.1 and gets its answers."""
    answer = {"type": "noul", "noul": 0.75}
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["content-length"]))))
            )
            data = json.dumps({"answers": {"q": answer}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for name in PROXY_VARS:
            monkeypatch.setenv(name, "http://127.0.0.1:9")
            monkeypatch.setenv(name.upper(), "http://127.0.0.1:9")
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
        # urlopen's shared opener reads the environment once; drop it so a regression would show.
        monkeypatch.setattr(h2o_parity.urllib.request, "_opener", None)
        bodies = {k: h2o_parity.TEXT_REQUESTS[k] for k in ("noul-outage", "noul-parcel")}
        url = f"http://127.0.0.1:{server.server_address[1]}"
        got = h2o_parity.from_url(url, bodies, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    assert got == {"noul-outage": answer, "noul-parcel": answer}
    assert [path for path, _ in seen] == ["/v1/systemone"] * 2
    assert seen[0][1]["state"] == bodies["noul-outage"]["state"]


# ---------------------------------------------------------------- run_checks.sh

import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

RUN_CHECKS = Path(__file__).resolve().parents[1] / "evaluation" / "h2o_lightning" / "run_checks.sh"


def _run_checks(*args, tmp_path):
    return subprocess.run(
        ["bash", str(RUN_CHECKS), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHON": sys.executable},
        cwd=tmp_path,
    )


def test_run_checks_takes_three_profiles_and_carries_no_cloud_or_credentials(tmp_path):
    """--help lists the gpu, cpu and mac profiles; another profile is refused; the script names
    no cloud provider, token or instance profile."""
    shown = _run_checks("--help", tmp_path=tmp_path)
    assert shown.returncode == 0
    for profile in ("--profile gpu", "--profile cpu", "--profile mac"):
        assert profile in shown.stdout
    refused = _run_checks("--profile", "cloud", tmp_path=tmp_path)
    assert refused.returncode != 0 and "gpu, cpu or mac" in refused.stderr
    text = RUN_CHECKS.read_text()
    assert not re.search(r"\baws\b", text, re.IGNORECASE)
    for word in ("AWS_", "HF_TOKEN", "instance-profile"):
        assert word not in text


def test_only_gguf_runs_the_gguf_rows_alone_and_an_unknown_group_is_refused(tmp_path):
    """`--profile gpu --only gguf` runs C-hf-gguf-source, C-hf (one-way only), C-gguf-Q8_0 and
    C-gguf-Q4_K_M, and compares C2-C4 and the record-only rows: no U, C1 or T row runs, and the
    header says only gguf. Python, curl, nvidia-smi and the llama-server are fakes here. An
    unknown group, or gguf with another profile, exits 2 with the usage."""
    bin_dir, files = tmp_path / "bin", tmp_path / "gguf"
    bin_dir.mkdir()
    files.mkdir()
    for name in ("h2o-lightning-4b-Q8_0.gguf", "h2o-lightning-4b-Q4_K_M.gguf"):
        (files / name).write_bytes(b"gguf")
    (files / "mmproj-lightning-f16.gguf").write_bytes(b"gguf")
    calls = tmp_path / "calls"
    fakes = {
        # every "$PY - ..." heredoc prints the GGUF folder (download); an --out file is written
        "py": '#!/bin/sh\necho "$*" >> "$CALLS"\nprev=""\nfor a in "$@"; do\n'
        '  [ "$prev" = --out ] && echo \'{"answers": {}}\' > "$a"\n  prev="$a"\ndone\n'
        '[ "$1" = - ] && { cat >/dev/null; echo "$FAKE_DIR"; }\nexit 0\n',
        "curl": "#!/bin/sh\nexit 0\n",
        "nvidia-smi": "#!/bin/sh\necho 46068\n",
        "llama-server": "#!/bin/sh\nexec sleep 30\n",
        "strands-decider": '#!/bin/sh\necho "strands-decider $*" >> "$CALLS"\nexit 1\n',
    }
    for name, body in fakes.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON": str(bin_dir / "py"),
        "CALLS": str(calls),
        "FAKE_DIR": str(files),
        "LLAMA_SERVER_BIN": str(bin_dir / "llama-server"),
    }
    out = tmp_path / "out"
    done = subprocess.run(
        ["bash", str(RUN_CHECKS), "--profile", "gpu", "--only", "gguf", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode in (0, 1), done.stderr
    results = (out / "results.md").read_text()
    assert "| only | gguf |" in results
    u = results.split("## U", 1)[1].split("\n## ", 1)[0]
    rows = [ln for ln in u.splitlines() if ln.startswith("| ") and not ln.startswith("| row |")]
    ran = [ln.split(" | ")[0][2:] for ln in rows]
    assert ran == ["C-hf-gguf-source", "C-hf", "C-gguf-Q8_0", "C-gguf-Q4_K_M"]
    assert "| C-hf | HF cuda answers, one-way only (v1.3.1" in u
    c = results.split("## C: answers compared", 1)[1].split("\n## ", 1)[0]
    compared = [ln[2:].split(" ")[0] for ln in c.splitlines() if ln.startswith("| C")]
    assert compared == ["C2", "C3", "C4", "C4", "C-gguf-Q8_0-vs-v1.3.1", "C-gguf-Q4_K_M-vs-v1.3.1"]
    assert "## T: not run (--only gguf)" in results
    log = calls.read_text()
    outs = re.findall(r"--out \S+/answers/(\S+)\.json", log)
    assert outs == ["hf-cuda-one-way-gguf-source", "hf-cuda-one-way", "gguf-Q8_0", "gguf-Q4_K_M"]
    for never in ("two-way", "bench_local.py", "docs_examples.py", "strands-decider ask"):
        assert never not in log, never

    for args in (("--profile", "gpu", "--only", "nope"), ("--profile", "cpu", "--only", "gguf")):
        refused = _run_checks(*args, tmp_path=tmp_path)
        assert refused.returncode == 2, args
        assert "--profile gpu|cpu|mac" in refused.stderr and "--only gguf" in refused.stderr
    assert (
        "unknown --only group: nope"
        in _run_checks("--profile", "gpu", "--only", "nope", tmp_path=tmp_path).stderr
    )


def test_run_checks_binds_every_server_it_starts_to_this_machine():
    """strands-decider serve and llama-server get --host 127.0.0.1; h2o's serve.sh (C1) gets
    SHIM_HOST=127.0.0.1 and no inherited VLLM_ARGS; no address listens on every interface."""
    text = RUN_CHECKS.read_text()
    assert "0.0.0.0" not in text
    assert re.search(r"strands-decider serve [^\n]*--host 127\.0\.0\.1", text)
    assert re.search(r"args=\([^\n]*--host 127\.0\.0\.1", text)  # llama-server
    official = text[text.index("start_official() {") : text.index("stop_official() {")]
    start = official.index('PATH="$venv/bin:$PATH"')
    launch = official[start : official.index('"$dir/serve.sh"', start)]
    assert "SHIM_HOST=127.0.0.1" in launch and 'VLLM_ARGS=""' in launch


def test_run_checks_talks_to_its_local_servers_without_environment_proxies():
    """Every curl line passes --noproxy '*'; the embedded Python HTTP uses an opener with an
    empty ProxyHandler and never the environment-reading urlopen."""
    lines = [ln for ln in RUN_CHECKS.read_text().splitlines() if re.search(r"\bcurl\b", ln)]
    calls = [ln for ln in lines if not ln.lstrip().startswith("#")]
    assert calls, "run_checks.sh is expected to poll its servers with curl"
    for ln in calls:
        assert "--noproxy '*'" in ln, ln
    text = RUN_CHECKS.read_text()
    assert "urllib.request.build_opener(urllib.request.ProxyHandler({}))" in text
    assert "urlopen(" not in text


def test_run_checks_pins_one_llama_cpp_build_that_ghcr_publishes():
    """Both llama.cpp containers use the same pinned build number, and it is not b11856, which
    ghcr.io/ggml-org/llama.cpp does not publish."""
    text = RUN_CHECKS.read_text()
    build = re.search(r'^LLAMA_CPP_BUILD="b(\d+)"$', text, re.MULTILINE)
    assert build, "run_checks.sh pins LLAMA_CPP_BUILD"
    assert re.search(
        r'^LLAMA_CPP_IMAGE_GPU="ghcr\.io/ggml-org/llama\.cpp:server-cuda-\$\{LLAMA_CPP_BUILD\}"$',
        text,
        re.MULTILINE,
    )
    assert re.search(
        r'^LLAMA_CPP_IMAGE_CPU="ghcr\.io/ggml-org/llama\.cpp:server-\$\{LLAMA_CPP_BUILD\}"$',
        text,
        re.MULTILINE,
    )
    assert build.group(1) != "11856"
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("11856" in ln for ln in code)  # only the pin's comment names it


def _pinned(text, name):
    found = re.search(rf'^{name}="\$\{{{name}:-([^}}]+)\}}"$', text, re.M)
    assert found, f"run_checks.sh pins {name}"
    return found.group(1)


def test_run_checks_serves_h2o_two_way_as_shipped_on_a_48_gb_gpu():
    """C1 TWO_WAY=1 runs serve.sh on stock vLLM with its own settings: no GPU_UTIL, MAX_IMAGES
    or MAX_PIXELS is passed, MAX_LEN only when OFFICIAL_TWO_WAY_MAX_LEN is set (empty by
    default), VLLM_ARGS stays empty, TWO_WAY=0 gets nothing, and nothing installs h2o's vLLM
    fork. The GPU bound is a 48 GB card's reported size (an L40S reports 46068 MiB)."""
    text = RUN_CHECKS.read_text()
    assert re.search(r'^OFFICIAL_TWO_WAY_MAX_LEN="\$\{OFFICIAL_TWO_WAY_MAX_LEN:-\}"$', text, re.M)
    bound = int(_pinned(text, "OFFICIAL_TWO_WAY_MIN_MIB"))
    assert 23034 < bound <= 46068  # an L4 is below it, an L40S above
    for gone in (
        "OFFICIAL_TWO_WAY_GPU_UTIL",
        "OFFICIAL_TWO_WAY_MAX_IMAGES",
        "OFFICIAL_TWO_WAY_MAX_PIXELS",
    ):
        assert gone not in text
    official = text[text.index("start_official() {") : text.index("stop_official() {")]
    settings = re.search(r'\[ "\$two_way" = 1 \] && ([^\n]*)', official)
    assert settings, "only TWO_WAY=1 may get a setting"
    assert settings.group(1) == (
        '[ -n "$OFFICIAL_TWO_WAY_MAX_LEN" ] && len=(MAX_LEN="$OFFICIAL_TWO_WAY_MAX_LEN")'
    )
    start = official.index('PATH="$venv/bin:$PATH"')
    launch = official[start : official.index('"$dir/serve.sh"', start)]
    assert '${len[@]+"${len[@]}"}' in launch and 'VLLM_ARGS=""' in launch
    assert "ONE_ENGINE" not in launch
    for var in ("MAX_LEN=", "GPU_UTIL=", "MAX_IMAGES=", "MAX_PIXELS="):
        assert var not in launch.replace('${len[@]+"${len[@]}"}', ""), var
    assert '"$venv/bin/pip" install -q "vllm==$VLLM_VERSION"' in official
    for gone in ("pip install git+", "git+https://", "OFFICIAL_ONE_ENGINE", "H2O_VLLM_FORK"):
        assert gone not in text


def test_c1_two_way_is_skipped_below_48_gb_with_its_reason_and_is_not_a_failure(tmp_path):
    """On a GPU under OFFICIAL_TWO_WAY_MIN_MIB (or with no nvidia-smi) two_way_skip gives the
    reason, naming the GPU's size; the gpu profile then records C1-official-1 and the C1 two-way
    comparison as SKIP with it, without starting serve.sh and without setting FAILED. On a 48 GB
    GPU it gives nothing and the row runs. C1 one-way runs either way."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    script = "\n".join(
        [
            "set -uo pipefail",
            "OFFICIAL_TWO_WAY_MIN_MIB=44000 FAILED=0 ROWS=()",
            _shell_functions("gpu_mib() {", "parity() {"),
            'r="$(two_way_skip)"; echo "reason=$r"',
            '[ -n "$r" ] && skip_row C1-official-1 "two-way" "$r"',
            'echo "failed=$FAILED skipped=$SKIPPED"; printf "%s\\n" "${ROWS[@]+"${ROWS[@]}"}"',
        ]
    )

    def run(mib):
        if mib is None:
            smi.unlink(missing_ok=True)
        else:
            smi.write_text(f"#!/bin/sh\necho '{mib}'\n")
            smi.chmod(0o755)
        env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"}
        done = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    small = run(23034)  # an L4
    assert (
        "reason=C1 two-way needs a GPU of 48 GB or more; this GPU has 22 GiB (23034 MiB)" in small
    )
    assert "failed=0 skipped=1" in small
    assert "| C1-official-1 | two-way | SKIP | - | C1 two-way needs a GPU of 48 GB" in small
    assert "this GPU has 0 GiB (0 MiB)" in run(None)
    large = run(46068)  # an L40S
    assert "reason=\n" in large and "failed=0 skipped=0" in large
    text = RUN_CHECKS.read_text()
    gpu = text[text.index("\ngpu)\n") : text.index("\ncpu)\n")]
    assert 'TWO_WAY_SKIP="$(two_way_skip)"' in gpu
    assert re.search(
        r'if \[ -z "\$TWO_WAY_SKIP" \]; then\n\s+row C1-official-1 [^\n]*c1_official 1\n'
        r'\s+else\n\s+skip_row C1-official-1 [^\n]*"\$TWO_WAY_SKIP"',
        gpu,
    )
    assert re.search(r'C_ROWS\+=\("\| C1 HF cuda two-way [^\n]*SKIP \(\$TWO_WAY_SKIP\) \|"\)', gpu)
    assert re.search(r"^  row C1-official-0 [^\n]*c1_official 0$", gpu, re.M)  # not inside the if
    assert "every row passed or was skipped" in text


def _shell_functions(start, end):
    text = RUN_CHECKS.read_text()
    return text[text.index(start) : text.index(end)]


def _fake_snapshot(tmp_path):
    """A Hub-cache-like snapshot: the two GGUF files are symlinks into blobs/."""
    blobs = tmp_path / "cache" / "blobs"
    snap = tmp_path / "cache" / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, blob in (("h2o-lightning-4b-Q8_0.gguf", "aaa"), ("mmproj.gguf", "bbb")):
        (blobs / blob).write_bytes(b"gguf")
        (snap / name).symlink_to(Path("..") / ".." / "blobs" / blob)
    return snap, blobs


def _start_llama(tmp_path, env_extra, out):
    script = "\n".join(
        [
            "set -uo pipefail",
            f'OUT="{out}" LLAMA_PORT=9 GGUF_REPO=r GGUF_REV=v GGUF_MMPROJ=mmproj.gguf',
            "LLAMA_CPP_IMAGE_GPU=img-gpu LLAMA_CPP_IMAGE_CPU=img-cpu",
            'download() { echo "$SNAP"; }',
            _shell_functions('LLAMA_PID="" LLAMA_CONTAINER=""', 'OFFICIAL_PID=""'),
            "start_llama Q8_0 gpu",
            "rc=$?",
            "stop_llama",  # as gguf_rows does after every quantisation
            'echo "rc=$rc"',
        ]
    )
    env = {**os.environ, **env_extra}
    for proxy in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        env.pop(proxy, None)
    began = time.monotonic()
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    return done, time.monotonic() - began


def test_llama_server_mounts_the_resolved_files_and_fails_at_once_when_it_stops(tmp_path):
    """The container gets the GGUF and the projector resolved with realpath (the cache's
    snapshot files are symlinks into blobs/), runs without --rm, and when it stops before /health
    answers the start fails at once (not after 600 s), writes `docker logs` to
    logs/llama-server-<quant>.log and removes the container. A LLAMA_SERVER_BIN process that
    dies fails the same way."""
    snap, blobs = _fake_snapshot(tmp_path)
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    calls = tmp_path / "docker-calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{calls}"\n'
        'case "$1" in\n'
        "run) echo 0123abcd ;;\n"
        "inspect) echo false ;;\n"
        "logs) echo 'gguf_init_from_file: failed to open GGUF file' ;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    done, took = _start_llama(
        tmp_path, {"PATH": f"{bin_dir}:{os.environ['PATH']}", "SNAP": str(snap)}, out
    )
    assert "rc=1" in done.stdout, done.stderr
    assert took < 30
    run = next(ln for ln in calls.read_text().splitlines() if ln.startswith("run "))
    assert "--rm" not in run.split()
    model, mmproj = (blobs / "aaa").resolve(), (blobs / "bbb").resolve()
    assert f"-v {model}:/models/h2o-lightning-4b-Q8_0.gguf:ro" in run
    assert f"-v {mmproj}:/models/mmproj.gguf:ro" in run
    assert str(snap) not in run
    log = (out / "logs" / "llama-server-Q8_0.log").read_text()
    assert "failed to open GGUF file" in log
    assert "logs/llama-server-Q8_0.log" in done.stderr
    run_at = next(i for i, ln in enumerate(calls.read_text().splitlines()) if ln.startswith("run "))
    assert any(ln.startswith("rm -f ") for ln in calls.read_text().splitlines()[run_at + 1 :])

    # LLAMA_SERVER_BIN: a server process that dies before /health answers fails at once too,
    # with its output kept in the same log; it was given the resolved files.
    server = tmp_path / "llama-server"
    server.write_text('#!/bin/sh\necho "args: $@"\necho "error: model load failed" >&2\nexit 1\n')
    server.chmod(0o755)
    done, took = _start_llama(tmp_path, {"LLAMA_SERVER_BIN": str(server), "SNAP": str(snap)}, out)
    assert "rc=1" in done.stdout, done.stderr
    assert took < 30
    log = (out / "logs" / "llama-server-Q8_0.log").read_text()
    assert "model load failed" in log
    assert f"-m {model}" in log
    assert "stopped before /health answered" in done.stderr


def test_official_failure_keeps_the_lines_around_its_first_error(tmp_path):
    """When h2o's serving fails, the lines around its first error go to
    logs/official-two-way-<n>-first-error.log, and the C1 rows point there. An OutOfMemory line
    is the error kept even after earlier ERROR lines, and when it ends a long traceback (vLLM's
    "EngineCore failed to start." then 40 and more frame lines) the excerpt starts at that
    traceback, so every frame down to the failing step is in it; an unrelated traceback earlier
    in the log, ended by other log lines, is not taken in. Without an OutOfMemory line the first
    Error / ERROR line is kept, with 60 lines before it."""
    log = tmp_path / "server.log"
    lines = [f"(EngineCore pid=1) INFO line {i}" for i in range(100)]
    lines[80] = "(EngineCore pid=1) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"
    lines[95] = "(APIServer pid=2) RuntimeError: Engine core initialization failed."
    # an unrelated traceback well before the OOM line, which other log lines end
    lines[10:14] = [
        "(EngineCore pid=1) Traceback (most recent call last):",
        '(EngineCore pid=1)   File "warmup.py", line 1, in <module>',
        "(EngineCore pid=1) UserWarning: unrelated",
        "(EngineCore pid=1) INFO after the unrelated traceback",
    ]
    log.write_text("\n".join(lines) + "\n")
    vllm = tmp_path / "vllm.log"
    e = "(EngineCore pid=3209) ERROR 10-29 21:50:24 [core.py:1366]"
    frames = [f'{e}   File "vllm/v1/worker/frame_{i}.py", line {i}, in step_{i}' for i in range(45)]
    frames[3] = f'{e}   File "vllm/v1/engine/core.py", line 308, in _initialize_kv_caches'
    frames[40] = f'{e}   File "vllm/v1/worker/gpu_model_runner.py", line 5000, in profile_run'
    early = [
        "(EngineCore pid=3209) WARNING 10-29 21:40:00 [utils.py:1] Traceback (most recent call last):",
        '(EngineCore pid=3209) WARNING 10-29 21:40:00 [utils.py:1]   File "probe.py", line 3, in x',
        "(EngineCore pid=3209) WARNING 10-29 21:40:00 [utils.py:1] UserWarning: no flash-attn",
    ]
    vllm.write_text(
        "\n".join(
            [f"(EngineCore pid=3209) INFO loading {i}" for i in range(20)]
            + early
            + [f"(EngineCore pid=3209) INFO loading {i}" for i in range(23, 100)]
            + [
                f"{e} EngineCore failed to start.",
                f"{e} Traceback (most recent call last):",
                *frames,
                f"{e} torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.",
                "(APIServer pid=3014) RuntimeError: Engine core initialization failed.",
            ]
        )
        + "\n"
    )
    script = "\n".join(
        [
            _shell_functions("traceback_start() {", "c1_log() {"),
            f'first_error "{log}" "{tmp_path}/first.log"; echo "rc=$?"',
            f'first_error "{tmp_path}/none.log" "{tmp_path}/none-first.log"; echo "rc=$?"',
            f'first_error "{vllm}" "{tmp_path}/vllm-first.log"; echo "rc=$?"',
        ]
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert done.stdout.split() == ["rc=0", "rc=1", "rc=0"], done.stderr
    kept = (tmp_path / "first.log").read_text()
    assert "OutOfMemoryError" in kept and "INFO line 20\n" in kept and "INFO line 19\n" not in kept
    assert "RuntimeError" in kept and "the first error is line 81" in kept
    assert "unrelated" not in kept
    kept = (tmp_path / "vllm-first.log").read_text()
    oom = 100 + 2 + 45 + 1  # the OutOfMemoryError's line
    assert f"lines 101-{oom + 30}; the first error is line {oom}" in kept
    assert kept.splitlines()[1] == f"{e} EngineCore failed to start."
    for line in ("Traceback (most recent call last)", "in _initialize_kv_caches", "in profile_run"):
        assert line in kept, line
    assert "torch.OutOfMemoryError: CUDA out of memory" in kept and "RuntimeError" in kept
    assert "INFO loading 99" not in kept and "no flash-attn" not in kept
    # without an OutOfMemory line: the first ERROR line, 60 lines before it
    vllm.write_text(vllm.read_text().replace("torch.OutOfMemoryError: CUDA out of memory.", "x"))
    first_error = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert first_error.returncode == 0
    kept = (tmp_path / "vllm-first.log").read_text()
    assert "the first error is line 101" in kept and "INFO loading 40\n" in kept
    text = RUN_CHECKS.read_text()
    official = text[text.index("c1_official() {") : text.index("first_error() {")]
    assert 'first_error "$log" "$OUT/logs/official-two-way-$1-first-error.log"' in official
    for n in ("1", "0"):
        assert re.search(rf'^\s+compare C1 .*"\$\(c1_log {n}\)"$', text, re.M), n


def test_gguf_rows_stop_with_the_reason_when_the_source_revision_cannot_be_read(tmp_path):
    """gguf_source_row, the gpu profile's first row, reads the HF build at the GGUF's source
    revision 9030a28 (torch, one-way). When that fails, the row and C2-C4 say STOPPED with the
    error line and the log, and FAILED stays 0; when it works, the row passes and C2-C4 compare
    as before. The GGUF vs v1.3.1 rows are "record only" either way and never set FAILED."""
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    a = tmp_path / "answers"
    a.mkdir()
    for name in ("gguf-Q8_0", "hf-cuda-one-way"):
        (a / f"{name}.json").write_text("{}")
    fake_py = tmp_path / "py"
    fake_py.write_text('#!/bin/sh\necho "| $9 | 15/16 | 0.2012 (limit 1) | FAIL |"\nexit 1\n')
    fake_py.chmod(0o755)
    text = RUN_CHECKS.read_text()

    def run(parity):
        script = "\n".join(
            [
                "set -uo pipefail",
                f'OUT="{out}" PY="{fake_py}" EVAL=e HF=h2oai/h2o-lightning-4b FAILED=0 ROWS=()',
                text[
                    text.index("GGUF_SOURCE_REV=") : text.index(
                        "\n", text.index("GGUF_SOURCE_REV=")
                    )
                ],
                _shell_functions("C_ROWS=()", "bench() {"),
                parity,
                "gguf_source_row",
                f'compare_gguf C2 "Q8_0 text" "{a}/none.json" "{a}/none.json" 0.05',
                f'compare_gguf C4 "Q8_0 images" "{a}/none.json" "{a}/none.json" 1',
                f'record C-gguf-Q8_0-vs-v1.3.1 "v1.3.1" "{a}/gguf-Q8_0.json" "{a}/hf-cuda-one-way.json"',
                f'record C-gguf-Q4_K_M-vs-v1.3.1 "v1.3.1" "{a}/none.json" "{a}/hf-cuda-one-way.json"',
                'echo "failed=$FAILED stopped=$STOPPED"',
                'printf "%s\\n" "${ROWS[@]}" "${C_ROWS[@]}"',
            ]
        )
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout

    stopped = run(
        'parity() { echo "Traceback (most recent call last):"; '
        'echo "KeyError: model.language_model.embed_tokens.weight"; return 1; }'
    )
    reason = (
        "the HF build at the GGUF's source revision 9030a28 could not be read: "
        "KeyError: model.language_model.embed_tokens.weight; logs/C-hf-gguf-source.log"
    )
    assert "failed=0 stopped=1" in stopped
    assert (
        "| C-hf-gguf-source | HF cuda one-way answers at the GGUF's source revision 9030a28 "
        "| STOPPED |"
    ) in stopped
    assert stopped.count(reason) == 3  # the row, C2 and C4
    assert f"| C2 Q8_0 text | - | - | STOPPED ({reason}) |" in stopped
    assert "| C-gguf-Q8_0-vs-v1.3.1 v1.3.1 | 15/16 | 0.2012 | record only |" in stopped
    assert (
        "| C-gguf-Q4_K_M-vs-v1.3.1 v1.3.1 | - | - | record only: could not compare (missing none.json"
        in stopped
    )
    assert "FAIL" not in stopped and "limit" not in stopped

    passed = run('parity() { echo "wrote answers"; }')
    assert "failed=1 stopped=0" in passed  # C2 and C4 compared: their answer files are missing
    assert "| C-hf-gguf-source | " in passed and " | PASS | " in passed
    assert "| C2 Q8_0 text | - | - | FAIL (missing none.json or none.json) |" in passed
    assert "| C-gguf-Q8_0-vs-v1.3.1 v1.3.1 | 15/16 | 0.2012 | record only |" in passed

    gpu = text[text.index("\ngpu)\n") : text.index("\ncpu)\n")]
    assert gpu.split("\n")[2].strip() == "gguf_source_row"  # the gpu profile's first row
    assert re.search(r"^  compare C[234] ", gpu, re.M) is None
    assert len(re.findall(r"^  compare_gguf C[234] ", gpu, re.M)) == 4
    assert 'record "C-gguf-$quant-vs-v1.3.1"' in gpu and '"$A/hf-cuda-one-way.json"' in gpu


def test_a_record_only_row_that_cannot_compare_keeps_its_id_and_says_why(tmp_path):
    """When h2o_parity.py cannot compare the two answer files (it stops with a message, or
    raises), the record-only row still has its ID and label and says "record only: could not
    compare (<the last line of its stderr>; logs/ID.log)", with "|" taken out of the reason and
    the whole stderr in logs/ID.log. FAILED stays 0."""
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    a = tmp_path / "answers"
    a.mkdir()
    for name in ("gguf-Q8_0", "hf-cuda-one-way"):
        (a / f"{name}.json").write_text("{}")
    stops = tmp_path / "stops"
    stops.write_text('#!/bin/sh\necho "q1: the two answers have different options" >&2\nexit 1\n')
    raises = tmp_path / "raises"
    raises.write_text(
        "#!/bin/sh\n"
        'echo "Traceback (most recent call last):" >&2\n'
        'echo "KeyError: a | b" >&2\n'
        "echo >&2\n"
        "exit 1\n"
    )
    for fake in (stops, raises):
        fake.chmod(0o755)
    other, ref = f"{a}/gguf-Q8_0.json", f"{a}/hf-cuda-one-way.json"
    script = "\n".join(
        [
            "set -uo pipefail",
            f'OUT="{out}" EVAL=e FAILED=0',
            _shell_functions("C_ROWS=()", "bench() {"),
            f'PY="{stops}"; record C-gguf-Q8_0-vs-v1.3.1 "v1.3.1" "{other}" "{ref}"',
            f'PY="{raises}"; record C-gguf-Q4_K_M-vs-v1.3.1 "v1.3.1" "{other}" "{ref}"',
            'echo "failed=$FAILED"',
            'printf "%s\\n" "${C_ROWS[@]}"',
        ]
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "failed=0" in done.stdout
    rows = done.stdout.splitlines()[1:]
    assert rows == [
        "| C-gguf-Q8_0-vs-v1.3.1 v1.3.1 | - | - | record only: could not compare "
        "(q1: the two answers have different options; logs/C-gguf-Q8_0-vs-v1.3.1.log) |",
        "| C-gguf-Q4_K_M-vs-v1.3.1 v1.3.1 | - | - | record only: could not compare "
        "(KeyError: a / b; logs/C-gguf-Q4_K_M-vs-v1.3.1.log) |",
    ]
    assert all(row.count("|") == 5 for row in rows)  # four columns
    log = (out / "logs" / "C-gguf-Q4_K_M-vs-v1.3.1.log").read_text()
    assert "Traceback (most recent call last)" in log and "KeyError: a | b" in log


def test_gguf_answer_rows_stop_with_the_source_revision_and_fail_on_the_llama_server_side(tmp_path):
    """C-gguf-<quant> takes its prompt settings from the GGUF's source revision 9030a28. When
    C-hf-gguf-source cannot read that revision, C-gguf-<quant> is not run and says STOPPED with
    the same reason, as C2-C4 do, and FAILED stays 0; the last line says the GGUF rows were
    STOPPED. When C-hf-gguf-source passes and C-gguf-<quant> fails on the llama-server's side,
    that row is a FAIL as before, FAILED is 1, and the last line says some rows FAILED (and
    STOPPED as well when there are both)."""
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    calls = tmp_path / "calls"
    fake_py = tmp_path / "py"
    fake_py.write_text("#!/bin/sh\nexit 0\n")
    fake_py.chmod(0o755)
    text = RUN_CHECKS.read_text()
    tail = text[text.index('stopped_as="') : text.index('exit "$FAILED"')]

    def run(source, stopped=None):
        script = "\n".join(
            [
                "set -uo pipefail",
                f'OUT="{out}" PY="{fake_py}" EVAL=e HERE=h DOCS=d HF=h2oai/h2o-lightning-4b',
                "FAILED=0 SKIPPED=0 ROWS=() REV=(--revision v1.3.1)",
                text[
                    text.index("GGUF_SOURCE_REV=") : text.index(
                        "\n", text.index("GGUF_SOURCE_REV=")
                    )
                ],
                _shell_functions("row() { # row ID", "download() {"),
                _shell_functions("C_ROWS=()", "bench() {"),
                _shell_functions("gguf_rows() {", "t_rows() {"),
                'say() { echo "$*"; }',
                "start_llama() { return 0; }",
                "stop_llama() { return 0; }",
                "u_cli_gguf() { return 0; }",
                "u_serve() { return 0; }",
                "bench_gguf() { return 0; }",
                "parity() {",
                f'  echo "$1" >> "{calls}"',
                '  case "$1" in',
                f"  hf-cuda-one-way-gguf-source) {source} ;;",
                '  *) echo "LlamaServerError: http://127.0.0.1:8080: connection refused"; return 1 ;;',
                "  esac",
                "}",
                "gguf_source_row",
                "gguf_rows U1b gpu T2 Q8_0 Q4_K_M",
                'G="$OUT/answers/hf-cuda-one-way-gguf-source"',
                'compare_gguf C2 "Q8_0 text" "$OUT/answers/gguf-Q8_0-text.json" "$G-text.json" 0.05',
                'compare_gguf C3 "Q4_K_M text" "$OUT/answers/gguf-Q4_K_M-text.json" "$G-text.json" 0.1',
                'compare_gguf C4 "Q8_0 images" "$OUT/answers/gguf-Q8_0-image.json" "$G-image.json" 1',
                f"STOPPED={stopped}" if stopped is not None else "",
                tail,
                'echo "failed=$FAILED stopped=$STOPPED"',
                'printf "%s\\n" "${ROWS[@]}" "${C_ROWS[@]}"',
            ]
        )
        calls.write_text("")
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout, calls.read_text().split()

    stopped, asked = run('echo "KeyError: serve_config.json: two_way"; return 1')
    reason = (
        "the HF build at the GGUF's source revision 9030a28 could not be read: "
        "KeyError: serve_config.json: two_way; logs/C-hf-gguf-source.log"
    )
    assert asked == ["hf-cuda-one-way-gguf-source"]  # no C-gguf-<quant> run
    assert "failed=0 stopped=1" in stopped
    for quant in ("Q8_0", "Q4_K_M"):
        assert f"| C-gguf-{quant} | GGUF {quant} answers | STOPPED | - | {reason} |" in stopped
    for c in ("C2 Q8_0 text", "C3 Q4_K_M text", "C4 Q8_0 images"):
        assert f"| {c} | - | - | STOPPED ({reason}) |" in stopped
    assert stopped.count(reason) == 6  # the source row, C-gguf twice, C2, C3 and C4
    assert "FAIL" not in stopped
    assert "every row passed; the GGUF rows were STOPPED" in stopped

    passed, asked = run('echo "wrote answers"')
    assert asked == ["hf-cuda-one-way-gguf-source", "gguf-Q8_0", "gguf-Q4_K_M"]
    assert "failed=1 stopped=0" in passed
    for quant in ("Q8_0", "Q4_K_M"):
        assert f"| C-gguf-{quant} | GGUF {quant} answers | FAIL | " in passed
    assert "LlamaServerError" in (out / "logs" / "C-gguf-Q8_0.log").read_text()
    assert "| C2 Q8_0 text | - | - | FAIL (missing " in passed and "STOPPED" not in passed
    assert passed.splitlines()[0].endswith(": some rows FAILED")

    # both: the last line names the FAIL rows and the STOPPED rows
    both, _ = run('echo "wrote answers"', stopped=1)
    assert both.splitlines()[0].endswith(
        ": some rows FAILED; the GGUF rows were STOPPED (the C-hf-gguf-source row gives the reason)"
    )


def test_run_checks_header_names_the_commit_and_the_pinned_revision(tmp_path):
    """--dry-run writes only the header, loading no model: the strands-decider commit and the
    pinned h2o revision e5c86b7 are in its first rows, on screen and in results.md."""
    done = _run_checks(
        "--profile", "cpu", "--out", str(tmp_path / "out"), "--dry-run", tmp_path=tmp_path
    )
    assert done.returncode == 0, done.stderr
    head = subprocess.run(
        ["git", "-C", str(RUN_CHECKS.parent), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    results = (tmp_path / "out" / "results.md").read_text()
    for text in (done.stdout, results):
        assert f"| strands-decider commit | {head} |" in text
        assert (
            "| h2o revision | h2oai/h2o-lightning-4b@e5c86b7564de9574fc48e759c226e50819d079ec"
            in text
        )
    assert "## U" not in results  # no row was run


# ---------------------------------------------------------------- the docs' examples

import ast  # noqa: E402
import inspect  # noqa: E402

from h2o_lightning import docs_examples  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs" / "h2o-lightning.md"


def _calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_the_docs_examples_compile_and_use_only_real_names():
    """The four marked examples compile, and every keyword they give load_engine,
    SystemOneRequest and the question types exists there (no weights are loaded)."""
    from strands_decider import schema
    from strands_decider.infer import load_engine

    examples = docs_examples.extract(DOCS.read_text())
    assert list(examples) == ["text", "image", "light-mlx", "light-llama"]
    params = set(inspect.signature(load_engine).parameters)
    models = {
        name: set(getattr(schema, name).model_fields)
        for name in ("SystemOneRequest", "ChoiceQuestion", "NoulQuestion", "ScoreQuestion")
    }
    for name, code in examples.items():
        compile(code, name, "exec")
        tree = ast.parse(code)
        [call] = _calls(tree, "load_engine")
        assert {k.arg for k in call.keywords} <= params, name
        assert _calls(tree, "SystemOneRequest"), name
        for model, fields in models.items():
            for c in _calls(tree, model):
                assert {k.arg for k in c.keywords} <= fields, (name, model)
    assert (
        ast.literal_eval(
            next(
                k.value
                for k in _calls(ast.parse(examples["light-llama"]), "load_engine")[0].keywords
                if k.arg == "engine"
            )
        )
        == "llama-server"
    )
    assert docs_examples.main(["--docs", str(DOCS), "--only", "nonexistent"]) == 1


def test_the_docs_list_the_requests_h2o_parity_sends_and_link_what_exists():
    """The 12 JSON requests in the docs are h2o_parity's, in order, and the docs' relative links
    and in-page anchors resolve."""
    text = DOCS.read_text()
    block = text.split("```json\n", 1)[1].split("```", 1)[0]
    assert [json.loads(line) for line in block.splitlines()] == list(
        h2o_parity.TEXT_REQUESTS.values()
    )
    for target in re.findall(r"\]\((\.\./[^)#]+)\)", text):
        assert (DOCS.parent / target).exists(), target
    anchors = {
        re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
        for h in re.findall(r"^#+ (.+)$", text, re.MULTILINE)
    }
    for anchor in re.findall(r"\]\(#([\w-]+)\)", text):
        assert anchor in anchors, anchor
