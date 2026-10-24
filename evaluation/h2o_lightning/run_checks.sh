#!/usr/bin/env bash
# Check h2o-lightning-4b through strands-decider on one machine, the way a user runs it, and write
# the result tables (evaluation/h2o_lightning/README.md).
#
#   run_checks.sh --profile gpu|cpu|mac [--out DIR] [--reference DIR] [--only GROUP] [--dry-run]
#
#   --profile gpu   a Linux machine with an NVIDIA GPU: U1, C1-C4, T1-T3
#   --profile cpu   a Linux machine without a GPU: U2, T7, T8
#   --profile mac   an Apple-silicon Mac: U3, C5, C6, T4-T6 (C5/C6 need --reference)
#   --out DIR       where the answers, timings and results.md go (default: ./h2o-checks)
#   --reference DIR a gpu run's --out; C5 and C6 compare with its reference/hf-cuda-one-way.json
#   --only gguf     with --profile gpu, only the GGUF rows: C-hf-gguf-source, C-hf (the v1.3.1
#                   one-way reference only), C-gguf-<quant>, C2-C4 and the record-only rows
#   --dry-run       write only the header (commit, versions, machine, pinned revisions)
#
# Runs where strands-decider is installed (pip install "strands-decider[vision] @ git+...@SHA",
# or "[vision,mlx]" on a Mac) or in a clone of the repository. Every model is a public Hub repo
# fetched anonymously at the revision pinned below; no credentials are read. The GGUF rows start a
# llama-server on 127.0.0.1: $LLAMA_SERVER_BIN when set, else the pinned llama.cpp container
# (docker); with neither, those rows fail (they are not skipped). Every row is run even when an
# earlier one fails; the exit code is 1 when any row failed.

set -uo pipefail

# ---------------------------------------------------------------- pinned revisions
H2O_REPO="h2oai/h2o-lightning-4b"
H2O_REV="e5c86b7564de9574fc48e759c226e50819d079ec" # v1.3.1
GGUF_REPO="h2oai/h2o-lightning-4b-gguf"
GGUF_REV="2504cec227e7df18d1764b6dff6ef37ed4666b1b" # the repo's tag v1.2.3
GGUF_MMPROJ="mmproj-lightning-f16.gguf"
# The GGUF files at GGUF_REV are quantised from h2o-lightning-4b at GGUF_SOURCE_REV (the GGUF
# repo's artifact_manifest_v3.json, source_model.revision). Its model.safetensors (sha256
# 2d51561f221e..., the weights of h2o's tags v1.2 to v1.2.3) is not v1.3.1's (0779980b4335...),
# and the GGUF repo has no files of v1.3.x (after v1.2.3 its next files are v1.4.0). C2-C4
# therefore compare the GGUF answers with the HF build at GGUF_SOURCE_REV, read one-way, and the
# GGUF rows of the C table take their prompt settings from that revision too (its
# serve_config.json one-way settings, tokenizer.json and chat_template.jinja are v1.3.1's).
GGUF_SOURCE_REV="9030a28dac81fcffd328f927e41a5deb85a27e5f"
MLX_Q8_REPO="netprojects/h2o-lightning-4b-mlx-q8"
MLX_Q8_REV="12a62e9c21e5630fbe16da1e52377d025800fe8e" # built from v1.3.1
MLX_Q4_REPO="netprojects/h2o-lightning-4b-mlx-q4"
MLX_Q4_REV="c807348cb7b747ad85bf8b7e6e1543eb9202ee0b" # built from v1.3.1
# llama.cpp b11854: ghcr.io/ggml-org/llama.cpp publishes both server-b11854 and server-cuda-b11854
# (b11856, the build the server API was read at, has no container). Between b11854 and b11856
# only CI and tests changed (ci/, .github/, tests/); tools/server/ and tools/mtmd/ are the same
# git trees (6e5fcbb, 09158c1), so /completion (n_probs, logit_bias, post_sampling_probs,
# completion_probabilities[0].top_probs id/prob), GET /props media_marker and the
# {prompt_string, multimodal_data} prompt are unchanged. b11870, the next build with both
# containers, rewrites the server entrypoint (#31113) and changes tools/mtmd image code, so it
# was not chosen. Both containers use the same build number.
LLAMA_CPP_BUILD="b11854"
LLAMA_CPP_IMAGE_GPU="ghcr.io/ggml-org/llama.cpp:server-cuda-${LLAMA_CPP_BUILD}"
LLAMA_CPP_IMAGE_CPU="ghcr.io/ggml-org/llama.cpp:server-${LLAMA_CPP_BUILD}"
# h2o's own serving (C1): stock vLLM and serve.sh at H2O_REV, which serves TWO_WAY=1 from one
# pooling-runner engine and TWO_WAY=0 from one generate-mode engine (no fork, no patch).
VLLM_VERSION="0.30.0"
# C1 two-way needs a GPU of 48 GB or more. serve.sh's TWO_WAY=1 engine runs with
# --no-enable-chunked-prefill, and vLLM 0.30.0 (engine/arg_utils.py, config/scheduler.py,
# multimodal/encoder_budget.py, profile_run in v1/worker/gpu_model_runner.py) then sets
# max_num_batched_tokens and the encoder cache budget to max(MAX_LEN, 2048) and, at start-up,
# profiles the vision tower on that many image tokens and one forward of that many tokens, with
# no memory cap. Memory is set by MAX_LEN (that profile) and GPU_UTIL (vLLM's share of the GPU).
# MAX_IMAGES is only the per-request image limit (--limit-mm-per-prompt): the profile's image
# count comes from the encoder budget, so MAX_IMAGES does not change the profile's size.
# On a 24 GB L4 the profile ran out of memory at serve.sh's MAX_LEN 40960, then at 8192
# ("Tried to allocate 256.00 MiB", 21.89 GiB in use), and at 2048 with GPU_UTIL 0.80 the engine
# still failed to start inside the profile (determine_available_memory; the kept excerpt ended
# before the exception line, which first_error below now keeps). So C1 two-way runs serve.sh as
# shipped (MAX_LEN 40960,
# GPU_UTIL 0.90, MAX_IMAGES 4, serve_config.json's MAX_PIXELS) on a 48 GB GPU (an L40S, a g6e
# instance), and is skipped, with the reason in the C table and not counted as a failure, on a
# smaller one. A 48 GB card reports less than 48 GiB (an L40S 46068 MiB, an RTX A6000 49140
# MiB), so the bound is 44000 MiB. OFFICIAL_TWO_WAY_MAX_LEN, when set, is passed as serve.sh's
# MAX_LEN; nothing else is changed, and VLLM_ARGS stays empty. C1 one-way (TWO_WAY=0) runs as
# shipped on any GPU, as before.
OFFICIAL_TWO_WAY_MIN_MIB="${OFFICIAL_TWO_WAY_MIN_MIB:-44000}"
OFFICIAL_TWO_WAY_MAX_LEN="${OFFICIAL_TWO_WAY_MAX_LEN:-}"
DECIDER_2B_REPO="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
DECIDER_2B_REV="216d1ccb46a9e9b036eb9187b248db98480f77d4"
DECIDER_E4B_REPO="StrandsAgents/strands-decider-E4B-gemma4-v1-2610" # Gemma 4: text only
DECIDER_E4B_REV="1344d44397c457f3be34b8ff15c593b41248c6bf"

PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$(cd "$HERE/.." && pwd)"
DOCS="${DOCS:-$EVAL/../docs/h2o-lightning.md}"
PORT="${PORT:-8790}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
OFFICIAL_PORT="${OFFICIAL_PORT:-8741}"
LENGTHS="${LENGTHS:-256,1024,4000}"
QUESTIONS="${QUESTIONS:-1,4}"
REPS="${REPS:-5}"
# ask, serve, bench_local, h2o_parity and the docs example all find the llama-server here.
export STRANDS_DECIDER_LLAMA_SERVER_URL="http://127.0.0.1:$LLAMA_PORT"

usage() {
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

PROFILE="" OUT="h2o-checks" REFERENCE="" ONLY="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
  --profile) PROFILE="${2:-}"; shift 2 ;;
  --out) OUT="${2:-}"; shift 2 ;;
  --reference) REFERENCE="${2:-}"; shift 2 ;;
  --only) ONLY="${2:-}"; shift 2 ;;
  --dry-run) DRY=1; shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
gpu | cpu | mac) ;;
*) echo "--profile must be gpu, cpu or mac" >&2; usage >&2; exit 2 ;;
esac
case "$ONLY" in
"") ;;
gguf)
  if [ "$PROFILE" != gpu ]; then
    echo "--only gguf goes with --profile gpu" >&2; usage >&2; exit 2
  fi
  ;;
*) echo "unknown --only group: $ONLY (the one group is gguf)" >&2; usage >&2; exit 2 ;;
esac
mkdir -p "$OUT"/answers "$OUT"/bench "$OUT"/logs "$OUT"/inputs "$OUT"/reference "$OUT"/details
OUT="$(cd "$OUT" && pwd)"
RESULTS="$OUT/results.md"
: >"$RESULTS"
FAILED=0

say() { printf '%s\n' "$*" | tee -a "$RESULTS"; }

# ---------------------------------------------------------------- header
strands_commit() {
  if git -C "$HERE" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$HERE" rev-parse HEAD
    return
  fi
  "$PY" - <<'PYEOF'
import json
from importlib import metadata

try:
    raw = metadata.distribution("strands-decider").read_text("direct_url.json")
    print(json.loads(raw).get("vcs_info", {}).get("commit_id", "unknown") if raw else "unknown")
except metadata.PackageNotFoundError:
    print("not installed")
PYEOF
}

versions() {
  "$PY" - <<'PYEOF'
from importlib import metadata

names = ["strands-decider", "torch", "transformers", "huggingface_hub", "mlx", "mlx-lm", "vllm"]
out = []
for name in names:
    try:
        out.append(f"{name} {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
print(", ".join(out))
PYEOF
}

machine() {
  local gpu="none" cpu mem
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
  fi
  if [ "$(uname -s)" = Darwin ]; then
    cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    mem="$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824)) GiB"
  else
    cpu="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //') x $(nproc 2>/dev/null)"
    mem="$(awk '/MemTotal/ {printf "%d GiB", $2 / 1048576}' /proc/meminfo 2>/dev/null)"
  fi
  printf '%s|%s, %s, GPU %s' "$(uname -a)" "$cpu" "$mem" "$gpu"
}

header() {
  local m
  m="$(machine)"
  say "# h2o-lightning checks: profile $PROFILE"
  say ""
  say "| item | value |"
  say "|---|---|"
  say "| strands-decider commit | $(strands_commit) |"
  say "| only | ${ONLY:-all rows} |"
  say "| packages | $("$PY" --version 2>&1), $(versions) |"
  say "| machine | ${m%%|*} |"
  say "| cpu, memory, gpu | ${m#*|} |"
  say "| h2o revision | $H2O_REPO@$H2O_REV (v1.3.1) |"
  say "| GGUF revision | $GGUF_REPO@$GGUF_REV (v1.2.3, quantised from $H2O_REPO@$GGUF_SOURCE_REV) |"
  say "| MLX revisions | $MLX_Q8_REPO@$MLX_Q8_REV, $MLX_Q4_REPO@$MLX_Q4_REV |"
  say "| llama.cpp | $LLAMA_CPP_BUILD (${LLAMA_SERVER_BIN:-container}) |"
  say "| h2o's own serving (C1) | stock vLLM $VLLM_VERSION, serve.sh at $H2O_REV as shipped (TWO_WAY=1${OFFICIAL_TWO_WAY_MAX_LEN:+ with MAX_LEN=$OFFICIAL_TWO_WAY_MAX_LEN}, which needs a GPU of 48 GB or more and is skipped below $OFFICIAL_TWO_WAY_MIN_MIB MiB; TWO_WAY=0) |"
  say "| strands-decider models | $DECIDER_2B_REPO@$DECIDER_2B_REV, $DECIDER_E4B_REPO@$DECIDER_E4B_REV |"
  say "| started | $(date -u +%Y-%m-%dT%H:%M:%SZ) |"
  say ""
}

header
if [ "$DRY" = 1 ]; then
  exit 0
fi

# ---------------------------------------------------------------- helpers
ROWS=()
selected() { # selected ID: the row is in the --only group (every row is without --only)
  case "${ONLY:-}" in
  "") return 0 ;;
  gguf)
    case "$1" in
    C-hf-gguf-source | C-hf | C-gguf-* | C2 | C3 | C4) return 0 ;;
    esac
    ;;
  esac
  return 1
}

row() { # row ID "description" command...: run it, time it, record PASS or FAIL
  local id="$1" what="$2" start status
  shift 2
  [ -z "${ONLY:-}" ] || selected "$id" || return 0
  echo "== $id: $what" >&2
  start=$(date +%s)
  if "$@" >"$OUT/logs/$id.log" 2>&1; then status=PASS; else status=FAIL; FAILED=1; fi
  ROWS+=("| $id | $what | $status | $(($(date +%s) - start)) s | logs/$id.log |")
}

download() { # download REPO REV [FILE...]: the local folder (public repos, no token)
  "$PY" - "$@" <<'PYEOF'
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download

repo, rev, *files = sys.argv[1:]
if files:
    for f in files:
        path = hf_hub_download(repo, f, revision=rev, token=False)
    print(os.path.dirname(path))
else:
    print(snapshot_download(repo, revision=rev, token=False, ignore_patterns=["assets/*"]))
PYEOF
}

make_image() { # a drawn PNG for --image
  "$PY" - "$OUT/inputs/shape.png" <<'PYEOF'
import sys

from PIL import Image, ImageDraw

img = Image.new("RGB", (448, 336), "white")
ImageDraw.Draw(img).ellipse((112, 84, 336, 252), fill="red")
img.save(sys.argv[1])
PYEOF
}

check_response() { # stdin: a /v1/systemone or `ask --json` reply that validates as SystemOneResponse
  "$PY" -c 'import json, sys
from strands_decider.schema import SystemOneResponse
d = json.load(sys.stdin); d.pop("latency_ms", None)
SystemOneResponse.model_validate(d); print("ok", list(d["answers"]))'
}

ASK_ARGS=(--state "Help! My payouts have been failing for 3 days!" --noul "Does this convey urgency?"
  --choice "Which team should handle this?=billing,sales,retail"
  --score "How frustrated is the writer?=calm,frustrated,depressed" --json)

u_cli() { # u_cli CHECKPOINT [ask options...]: one `ask`, its answer validated
  local ckpt="$1"
  shift
  strands-decider ask "$ckpt" "${ASK_ARGS[@]}" "$@" | check_response
}

u_cli_readings() { # u_cli_readings DEVICE READING...: text and image `ask` per reading
  local device="$1" a
  shift
  for a in "$@"; do
    u_cli "$HF" "${REV[@]}" --device "$device" --attention "$a" || return 1
    u_cli "$HF" "${REV[@]}" --device "$device" --attention "$a" --image "$IMG" || return 1
  done
}

u_cli_gguf() {
  u_cli "$HF" "${REV[@]}" --engine llama-server && u_cli "$HF" "${REV[@]}" --engine llama-server --image "$IMG"
}

u_serve() { # u_serve IMAGES(0|1) CHECKPOINT [serve options...]: start serve, POST once, stop
  local images="$1" ckpt="$2" pid rc=1
  shift 2
  strands-decider serve "$ckpt" --host 127.0.0.1 --port "$PORT" "$@" &
  pid=$!
  for _ in $(seq 1 900); do
    curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if "$PY" - "$PORT" "$images" "$IMG" <<'PYEOF' | check_response; then rc=0; fi
import base64
import json
import sys
import urllib.request

port, images, image = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
body = {
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "", "sales": ""}},
    },
}
if images:
    with open(image, "rb") as fh:
        body["images"] = [base64.b64encode(fh.read()).decode()]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"content-type": "application/json"}
)
# serve is on this machine: never go through a proxy from the environment.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
print(opener.open(req, timeout=600).read().decode())
PYEOF
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  return $rc
}

LLAMA_PID="" LLAMA_CONTAINER="" LLAMA_QUANT=""
start_llama() { # start_llama QUANT gpu|cpu: llama-server with the projector on 127.0.0.1:$LLAMA_PORT
  local quant="$1" kind="$2" dir model mmproj
  local args=(-c 40960 -np 1 --no-kv-unified --image-max-tokens 1024 --host 127.0.0.1 --port "$LLAMA_PORT")
  stop_llama
  LLAMA_QUANT="$quant"
  dir="$(download "$GGUF_REPO" "$GGUF_REV" "h2o-lightning-4b-$quant.gguf" "$GGUF_MMPROJ")" || return 1
  # hf_hub_download returns the cache's snapshot path, whose files are symlinks to
  # ../../blobs/<hash>: a mount of the snapshot directory alone leaves them dangling inside the
  # container. Resolve each file to its blob with realpath and give the container those two
  # files, read-only and under their own names, and nothing else of the cache. This works for a
  # cache with or without symlinks and keeps no second copy of the weights.
  model="$(realpath "$dir/h2o-lightning-4b-$quant.gguf")" || return 1
  mmproj="$(realpath "$dir/$GGUF_MMPROJ")" || return 1
  [ "$kind" = gpu ] && args+=(-ngl 999)
  if [ -n "${LLAMA_SERVER_BIN:-}" ]; then
    "$LLAMA_SERVER_BIN" -m "$model" --mmproj "$mmproj" "${args[@]}" >"$OUT/logs/llama-server-$quant.log" 2>&1 &
    LLAMA_PID=$!
  elif command -v docker >/dev/null 2>&1; then
    local image="$LLAMA_CPP_IMAGE_CPU" gpus=()
    [ "$kind" = gpu ] && image="$LLAMA_CPP_IMAGE_GPU" && gpus=(--gpus all)
    LLAMA_CONTAINER="h2o-checks-llama-$$"
    docker rm -f "$LLAMA_CONTAINER" >/dev/null 2>&1
    # No --rm: a container that stops keeps its output for `docker logs`; stop_llama removes it.
    docker run -d --name "$LLAMA_CONTAINER" ${gpus[@]+"${gpus[@]}"} --network host \
      -v "$model:/models/h2o-lightning-4b-$quant.gguf:ro" -v "$mmproj:/models/$GGUF_MMPROJ:ro" \
      "$image" -m "/models/h2o-lightning-4b-$quant.gguf" --mmproj "/models/$GGUF_MMPROJ" "${args[@]}" || {
      llama_failed "docker run failed"
      return 1
    }
  else
    echo "no llama-server: set LLAMA_SERVER_BIN or install docker" >&2
    return 1
  fi
  for _ in $(seq 1 600); do
    curl -sf --noproxy '*' "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1 && return 0
    llama_running || { llama_failed "llama-server stopped before /health answered"; return 1; }
    sleep 1
  done
  llama_failed "llama-server did not become healthy in 600 s"
  return 1
}

llama_running() { # the llama-server process, or its container, is still running
  if [ -n "$LLAMA_PID" ]; then
    kill -0 "$LLAMA_PID" 2>/dev/null
  else
    [ "$(docker inspect -f '{{.State.Running}}' "$LLAMA_CONTAINER" 2>/dev/null)" = true ]
  fi
}

llama_logs() { # the container's output to logs/llama-server-QUANT.log (a binary writes there itself)
  if [ -n "$LLAMA_CONTAINER" ]; then
    docker logs --tail 300 "$LLAMA_CONTAINER" >"$OUT/logs/llama-server-$LLAMA_QUANT.log" 2>&1
  fi
  return 0
}

llama_failed() { # llama_failed REASON: say why, with the server's last lines
  llama_logs
  echo "$1; the server's output is in logs/llama-server-$LLAMA_QUANT.log, which ends:" >&2
  tail -n 30 "$OUT/logs/llama-server-$LLAMA_QUANT.log" >&2 2>/dev/null
}

stop_llama() {
  if [ -n "$LLAMA_PID" ]; then kill "$LLAMA_PID" 2>/dev/null; wait "$LLAMA_PID" 2>/dev/null; fi
  if [ -n "$LLAMA_CONTAINER" ]; then
    llama_logs
    docker rm -f "$LLAMA_CONTAINER" >/dev/null 2>&1
  fi
  LLAMA_PID="" LLAMA_CONTAINER=""
  return 0
}

OFFICIAL_PID=""
start_official() { # start_official 1|0: h2o's serve.sh (TWO_WAY=1|0) in its own venv, stock vLLM
  local two_way="$1" venv="$OUT/official-venv" dir len=()
  stop_official
  if [ ! -x "$venv/bin/vllm" ]; then
    "$PY" -m venv "$venv" || return 1
    "$venv/bin/pip" install -q "vllm==$VLLM_VERSION" || return 1
  fi
  dir="$(download "$H2O_REPO" "$H2O_REV")" || return 1
  # serve.sh's own settings for both readings (see OFFICIAL_TWO_WAY_MIN_MIB above); TWO_WAY=1 gets
  # MAX_LEN only when OFFICIAL_TWO_WAY_MAX_LEN is set.
  [ "$two_way" = 1 ] && [ -n "$OFFICIAL_TWO_WAY_MAX_LEN" ] && len=(MAX_LEN="$OFFICIAL_TWO_WAY_MAX_LEN")
  # serve.sh at H2O_REV binds both vLLM engines with a fixed `--host 127.0.0.1` and the shim to
  # $SHIM_HOST (default 127.0.0.1). Pin SHIM_HOST and empty VLLM_ARGS (extra `vllm serve` flags,
  # a later --host would win), so nothing inherited can make either listen beyond this machine.
  # Its own process group, so stopping it stops every vLLM it started and nothing else.
  env PATH="$venv/bin:$PATH" TWO_WAY="$two_way" PORT="$OFFICIAL_PORT" VLLM_PORT=$((OFFICIAL_PORT + 1)) \
    SHIM_HOST=127.0.0.1 VLLM_ARGS="" ${len[@]+"${len[@]}"} \
    setsid bash "$dir/serve.sh" >"$OUT/logs/official-two-way-$two_way-server.log" 2>&1 &
  OFFICIAL_PID=$!
  for _ in $(seq 1 1800); do
    curl -sf --noproxy '*' "http://127.0.0.1:$OFFICIAL_PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$OFFICIAL_PID" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

stop_official() {
  if [ -n "$OFFICIAL_PID" ]; then
    kill -- "-$OFFICIAL_PID" 2>/dev/null
    wait "$OFFICIAL_PID" 2>/dev/null
  fi
  OFFICIAL_PID=""
  return 0
}
trap 'stop_llama; stop_official' EXIT

gpu_mib() { # the first GPU's memory in MiB, 0 without nvidia-smi
  local mib
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
  echo "${mib:-0}"
}

two_way_skip() { # why C1 two-way is skipped on this GPU; nothing when it runs
  local mib
  mib="$(gpu_mib)"
  if [ "$mib" -lt "$OFFICIAL_TWO_WAY_MIN_MIB" ]; then
    echo "C1 two-way needs a GPU of 48 GB or more; this GPU has $((mib / 1024)) GiB ($mib MiB)"
  fi
}

SKIPPED=0
skip_row() { # skip_row ID "description" REASON: a row that is not run, and not a failure
  [ -z "${ONLY:-}" ] || selected "$1" || return 0
  ROWS+=("| $1 | $2 | SKIP | - | $3 |")
  SKIPPED=1
}

parity() { # parity NAME [h2o_parity options...]: answers to answers/NAME.json
  local name="$1"
  shift
  "$PY" "$EVAL/h2o_parity.py" "$@" --out "$OUT/answers/$name.json"
}

c_reference() { # with --only gguf, the v1.3.1 one-way reference only (the record-only rows')
  parity hf-cuda-one-way "$HF" "${REV[@]}" --device cuda --attention one-way &&
    { [ "${ONLY:-}" = gguf ] || parity hf-cuda-two-way "$HF" "${REV[@]}" --device cuda --attention two-way; } &&
    cp "$OUT/answers/hf-cuda-one-way.json" "$OUT/reference/hf-cuda-one-way.json"
}

c1_official() { # c1_official 1|0
  local rc log="$OUT/logs/official-two-way-$1-server.log"
  rm -f "$OUT/logs/official-two-way-$1-first-error.log" # not one left from an earlier run
  start_official "$1" && parity "official-two-way-$1" --url "http://127.0.0.1:$OFFICIAL_PORT"
  rc=$?
  stop_official
  if [ "$rc" != 0 ] && first_error "$log" "$OUT/logs/official-two-way-$1-first-error.log"; then
    echo "h2o's serving failed; the lines around its first error (logs/official-two-way-$1-first-error.log):"
    cat "$OUT/logs/official-two-way-$1-first-error.log"
  fi
  return $rc
}

traceback_start() { # traceback_start LOG N: the first line of the traceback that line N ends, or 0
  # Walks back from line N-1 while the lines are a Python traceback's (frames and their indented
  # source, ^^^ markers, chained exceptions), once the prefixes a vLLM log puts on every line are
  # dropped ("(EngineCore pid=N) " and "ERROR 10-29 21:50:24 [core.py:1366] "); it stops at the
  # first other line, so an earlier traceback that the kept line is not part of is never reached.
  awk -v n="$2" '
    function body(s) {
      sub(/^\([^)]*\) /, "", s)
      sub(/^(DEBUG|INFO|WARNING|ERROR|CRITICAL) [0-9-]+ [0-9:.]+ \[[^]]*\] ?/, "", s)
      return s
    }
    function chain(s) {
      return s ~ /^During handling of the above exception/ || s ~ /^The above exception was the direct cause/
    }
    NR <= n { l[NR] = body($0) }
    END {
      t = 0
      for (i = n - 1; i >= 1; i--) {
        s = l[i]
        if (s ~ /^Traceback \(most recent call last\)/) {
          t = i
          if (chain(l[i - 1]) || (l[i - 1] == "" && i > 2 && chain(l[i - 2]))) continue
          break
        }
        if (s ~ /^[ \t]/ || chain(s)) continue
        # an earlier exception of a chain: its line, then a blank line and "During handling ..."
        if (t > 0 && s ~ /^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Exit|Interrupt)(:|$)/ &&
            (l[i + 1] == "" || chain(l[i + 1]))) continue
        if (s == "" && ((i > 1 && chain(l[i - 1])) || chain(l[i + 1]))) continue
        break
      }
      print t
    }' "$1"
}

first_error() { # first_error LOG OUT: the lines around LOG's first error, to OUT
  # The line to keep is the first OutOfMemory / out of memory line, else the first Error / ERROR
  # line, with the 60 lines before it. vLLM 0.30.0 reports a failed start as "ERROR ...
  # EngineCore failed to start." and then the traceback (v1/engine/core.py logger.exception),
  # whose last line is the OutOfMemoryError: when the kept line ends a traceback, the excerpt
  # starts at that traceback (or at the "failed to start" line just before it) instead, so every
  # frame down to the failing step is kept however long the traceback is. Only the traceback the
  # kept line ends counts (traceback_start): an unrelated traceback earlier in the log does not.
  local n t s from
  n="$(grep -n -m1 -E 'OutOfMemory|out of memory' "$1" 2>/dev/null | cut -d: -f1)"
  [ -n "$n" ] || n="$(grep -n -m1 -E 'Error|ERROR' "$1" 2>/dev/null | cut -d: -f1)"
  [ -n "$n" ] || return 1
  from=$((n > 60 ? n - 60 : 1))
  t="$(traceback_start "$1" "$n")"
  if [ "$t" -gt 0 ]; then
    s="$(head -n "$t" "$1" | grep -n 'failed to start' | tail -n 1 | cut -d: -f1)"
    [ -n "$s" ] && [ $((t - s)) -le 5 ] && t="$s"
    from="$t"
  fi
  {
    echo "# $(basename "$1"), lines $from-$((n + 30)); the first error is line $n"
    sed -n "$from,$((n + 30))p" "$1"
  } >"$2"
}

c1_log() { # c1_log 1|0: the log a C1 row points at
  if [ -f "$OUT/logs/official-two-way-$1-first-error.log" ]; then
    echo "logs/official-two-way-$1-first-error.log"
  else
    echo "logs/official-two-way-$1-server.log"
  fi
}

split_answers() { # split_answers FILE: FILE-text.json and FILE-image.json beside it
  "$PY" - "$1" <<'PYEOF'
import json
import sys

path = sys.argv[1]
with open(path) as fh:
    answers = json.load(fh)["answers"]
for part, images in (("text", False), ("image", True)):
    keep = {k: v for k, v in answers.items() if k.startswith("image-") == images}
    with open(path[: -len(".json")] + f"-{part}.json", "w") as fh:
        json.dump({"answers": keep}, fh)
PYEOF
}

C_ROWS=()
# C2-C4 and the record-only rows also keep their comparison per request (h2o_parity.py
# --details): one markdown table each, in results.md after the C table, and every request's line
# on the console (stderr) as "CHECK-DIFF <row> <answers file> | request | ... |". A row that
# could not compare (STOPPED, a missing file, "could not compare") has none.
BD_IDS=() BD_LABELS=() BD_OTHERS=() BD_REFS=() BD_FILES=()
breakdown_file() { # breakdown_file ID OTHER: where h2o_parity.py writes the rows per request
  mkdir -p "$OUT/details"
  echo "$OUT/details/$1-$(basename "$2" .json).md"
}

add_breakdown() { # add_breakdown ID LABEL OTHER REF FILE: keep FILE when h2o_parity.py wrote it
  [ -s "$5" ] || return 0
  BD_IDS+=("$1") BD_LABELS+=("$2") BD_OTHERS+=("$3") BD_REFS+=("$4") BD_FILES+=("$5")
}

breakdown_section() { # the comparisons per request, to results.md (say) and the console
  local i line stem
  [ "${#BD_IDS[@]}" -gt 0 ] || return 0
  say "## C: answers per request (C2-C4 and the record-only rows)"
  say ""
  for i in "${!BD_IDS[@]}"; do
    stem="$(basename "${BD_OTHERS[$i]}" .json)"
    say "### ${BD_IDS[$i]} ${BD_LABELS[$i]}"
    say ""
    say "Answers \`$(basename "${BD_OTHERS[$i]}")\` against the reference \`$(basename "${BD_REFS[$i]}")\`."
    say ""
    while IFS= read -r line; do
      say "$line"
      case "$line" in
      "| request |"* | "|---"*) ;;
      *) echo "CHECK-DIFF ${BD_IDS[$i]} $stem $line" >&2 ;;
      esac
    done <"${BD_FILES[$i]}"
    say ""
  done
}

compare() { # compare ID LABEL OTHER REF MAX_DP [LOG]: one row of the C table (LOG: where to look)
  # With C_BREAKDOWN=1 (compare_gguf), the comparison per request is kept as well.
  local id="$1" label="$2" other="$3" ref="$4" max="$5" log="${6:-}" line details=""
  [ -z "${ONLY:-}" ] || selected "$id" || return 0
  if [ -f "$other" ] && [ -f "$ref" ]; then
    if [ "${C_BREAKDOWN:-0}" = 1 ]; then
      details="$(breakdown_file "$id" "$other")"
      rm -f "$details"
    fi
    line="$("$PY" "$EVAL/h2o_parity.py" --answers "$other" --compare "$ref" --max-dp "$max" --label "$id $label" ${details:+--details "$details"})"
    case "$line" in *"| PASS |") ;; *) FAILED=1 ;; esac
    [ -z "$details" ] || add_breakdown "$id" "$label" "$other" "$ref" "$details"
  else
    line="| $id $label | - | - | FAIL (missing $(basename "$other") or $(basename "$ref"))${log:+; see $log} |"
    FAILED=1
  fi
  C_ROWS+=("$line")
}

# The GGUF rows' reference is the HF build at GGUF_SOURCE_REV, the revision the GGUF files are
# quantised from, so the C table measures quantisation on the same weights. gguf_source_row,
# the gpu profile's first row, checks that this strands-decider reads that revision (torch,
# one-way; the revision has no label_rows.lightning, which only two-way needs). When it cannot,
# the row, C-gguf-<quant> (whose prompt settings come from that revision) and C2-C4 are STOPPED
# with the reason, which is not a failure: without the weights the GGUF was made from there is
# nothing to measure its quantisation against. When it can, a C-gguf-<quant> row that fails (the
# llama-server's side) is a FAIL, and so are C2-C4 without its answers.
GGUF_SOURCE_STOP="" STOPPED=0
error_line() { # error_line LOG: the log's last error line (else its last line), for a table cell
  local line
  line="$(grep -E 'Error|error|Exception' "$1" 2>/dev/null | tail -n 1)"
  [ -n "$line" ] || line="$(grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1)"
  line="${line//|//}"
  echo "${line:0:200}"
}

gguf_source_row() { # gguf_source_row: read the GGUF's source revision, or stop C2-C4 with why
  local id=C-hf-gguf-source start
  local what="HF cuda one-way answers at the GGUF's source revision ${GGUF_SOURCE_REV:0:7}"
  echo "== $id: $what" >&2
  start=$(date +%s)
  if parity hf-cuda-one-way-gguf-source "$HF" --revision "$GGUF_SOURCE_REV" --device cuda \
    --attention one-way >"$OUT/logs/$id.log" 2>&1; then
    ROWS+=("| $id | $what | PASS | $(($(date +%s) - start)) s | logs/$id.log |")
  else
    GGUF_SOURCE_STOP="the HF build at the GGUF's source revision ${GGUF_SOURCE_REV:0:7} could not be read: $(error_line "$OUT/logs/$id.log"); logs/$id.log"
    ROWS+=("| $id | $what | STOPPED | $(($(date +%s) - start)) s | $GGUF_SOURCE_STOP |")
    STOPPED=1
  fi
}

compare_gguf() { # compare_gguf ID LABEL OTHER REF MAX_DP [LOG]: compare, or STOPPED with why
  if [ -n "$GGUF_SOURCE_STOP" ]; then
    C_ROWS+=("| $1 $2 | - | - | STOPPED ($GGUF_SOURCE_STOP) |")
  else
    C_BREAKDOWN=1 compare "$@"
  fi
}

record() { # record ID LABEL OTHER REF: a C row kept for the record only, never PASS or FAIL
  # h2o_parity.py exits 0 (PASS) or 1 (FAIL) with the row on stdout; when it cannot compare (it
  # stops with a message, or raises) there is no row, so the row says why, with the last line of
  # its stderr, kept in logs/ID.log. Either way FAILED is not set.
  local log="$OUT/logs/$1.log" line rc reason details
  [ -z "${ONLY:-}" ] || selected "$1" || return 0
  if [ -f "$3" ] && [ -f "$4" ]; then
    details="$(breakdown_file "$1" "$3")"
    rm -f "$details"
    line="$("$PY" "$EVAL/h2o_parity.py" --answers "$3" --compare "$4" --max-dp 1 --label "$1 $2" --details "$details" 2>"$log")"
    rc=$?
    if [ "$rc" -le 1 ] && [[ "$line" == "| $1 $2 |"*" |" ]]; then
      line="${line% | PASS |}"
      line="${line% | FAIL |}"
      line="${line/ (limit 1)/} | record only |"
      add_breakdown "$1" "$2" "$3" "$4" "$details"
    else
      reason="$(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -n 1)"
      reason="${reason//|//}"
      reason="${reason:0:200}"
      line="| $1 $2 | - | - | record only: could not compare (${reason:-h2o_parity.py exited $rc}; logs/$1.log) |"
    fi
  else
    line="| $1 $2 | - | - | record only: could not compare (missing $(basename "$3") or $(basename "$4")) |"
  fi
  C_ROWS+=("$line")
}

bench() { # bench ID [bench_local options...]
  local id="$1"
  shift
  "$PY" "$EVAL/bench_local.py" "$@" --lengths "$LENGTHS" --questions "$QUESTIONS" --reps "$REPS" \
    --out "$OUT/bench/$id.csv"
}

bench_hf() { # bench_hf ID DEVICE READING...: text and image per reading
  local id="$1" device="$2" a
  shift 2
  for a in "$@"; do
    bench "$id-$a-text" "$HF" "${REV[@]}" --device "$device" --attention "$a" || return 1
    bench "$id-$a-image" "$HF" "${REV[@]}" --device "$device" --attention "$a" --image "$IMG" || return 1
  done
}

bench_gguf() { # bench_gguf ID
  bench "$1-text" "$HF" "${REV[@]}" --engine llama-server &&
    bench "$1-image" "$HF" "${REV[@]}" --engine llama-server --image "$IMG"
}

bench_deciders() { # bench_deciders ID DEVICE: 2B text and image, E4B text (Gemma 4 reads no images)
  bench "$1-2B-text" "$DECIDER_2B_REPO" --revision "$DECIDER_2B_REV" --device "$2" &&
    bench "$1-2B-image" "$DECIDER_2B_REPO" --revision "$DECIDER_2B_REV" --device "$2" --image "$IMG" &&
    bench "$1-E4B-text" "$DECIDER_E4B_REPO" --revision "$DECIDER_E4B_REV" --device "$2"
}

gguf_rows() { # gguf_rows PREFIX KIND T_ID QUANT...: per quantisation, start llama-server and run its rows
  local prefix="$1" kind="$2" tid="$3" quant first=1
  shift 3
  for quant in "$@"; do
    if start_llama "$quant" "$kind" >"$OUT/logs/start-llama-$quant.log" 2>&1; then
      if [ "$kind" = gpu ]; then
        # The GGUF answers take their prompt settings from GGUF_SOURCE_REV: when C-hf-gguf-source
        # could not read that revision, this row is STOPPED with the same reason, not run.
        if [ -n "$GGUF_SOURCE_STOP" ]; then
          ROWS+=("| C-gguf-$quant | GGUF $quant answers | STOPPED | - | $GGUF_SOURCE_STOP |")
        else
          row "C-gguf-$quant" "GGUF $quant answers" parity "gguf-$quant" "$HF" --revision "$GGUF_SOURCE_REV" --engine llama-server
        fi
      fi
      if [ "$first" = 1 ]; then
        row "$prefix-python" "docs example light-llama ($quant)" "$PY" "$HERE/docs_examples.py" --docs "$DOCS" --only light-llama
        row "$prefix-cli" "ask --engine llama-server, text and image ($quant)" u_cli_gguf
        row "$prefix-serve" "serve --engine llama-server, image request ($quant)" u_serve 1 "$HF" "${REV[@]}" --engine llama-server
      fi
      row "$tid-$quant" "time GGUF $quant, text and image" bench_gguf "$tid-$quant"
    else
      ROWS+=("| $prefix-$quant | llama-server $quant did not start | FAIL | - | logs/start-llama-$quant.log, logs/llama-server-$quant.log |")
      FAILED=1
    fi
    stop_llama
    first=0
  done
}

t_rows() { # the T table from every bench CSV
  "$PY" - "$OUT/bench" <<'PYEOF'
import csv
import glob
import os
import sys

print("| row | device | dtype | images | state tokens | questions | path | median ms | p90 ms |")
print("|---|---|---|---|---|---|---|---|---|")
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.csv"))):
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            print(
                f"| {os.path.basename(path)[:-4]} | {r['device']} | {r['dtype']} | {r.get('images') or 0} | "
                f"{r['state_tokens']} | {r['questions']} | {r['path']} | {r['median_ms']} | {r['p90_ms']} |"
            )
PYEOF
}

make_image
IMG="$OUT/inputs/shape.png"
HF="$H2O_REPO"
REV=(--revision "$H2O_REV")
A="$OUT/answers"

# ---------------------------------------------------------------- profiles
case "$PROFILE" in
gpu)
  gguf_source_row
  C_HF_WHAT="HF cuda answers, one-way and two-way"
  [ "$ONLY" = gguf ] && C_HF_WHAT="HF cuda answers, one-way only (v1.3.1, the record-only rows' reference)"
  row U1a-python "docs examples text and image (cuda)" "$PY" "$HERE/docs_examples.py" --docs "$DOCS" --only text,image
  row U1a-cli "ask two-way and one-way, text and image (cuda)" u_cli_readings cuda two-way one-way
  row U1a-serve "serve --vision, image request (cuda)" u_serve 1 "$HF" "${REV[@]}" --device cuda --vision
  row C-hf "$C_HF_WHAT" c_reference
  TWO_WAY_SKIP="$(two_way_skip)"
  if [ -z "$TWO_WAY_SKIP" ]; then
    row C1-official-1 "h2o's own serving, TWO_WAY=1" c1_official 1
  else
    skip_row C1-official-1 "h2o's own serving, TWO_WAY=1" "$TWO_WAY_SKIP"
  fi
  row C1-official-0 "h2o's own serving, TWO_WAY=0" c1_official 0
  gguf_rows U1b gpu T2 Q8_0 Q4_K_M
  row T1 "time HF build, two-way and one-way, text and image (cuda)" bench_hf T1 cuda two-way one-way
  row T3 "time strands-decider-2B (text, image) and E4B (text) (cuda)" bench_deciders T3 cuda
  for f in hf-cuda-one-way hf-cuda-one-way-gguf-source gguf-Q8_0 gguf-Q4_K_M; do [ -f "$A/$f.json" ] && split_answers "$A/$f.json"; done
  if [ -z "$TWO_WAY_SKIP" ]; then
    compare C1 "HF cuda two-way vs h2o TWO_WAY=1 (answers; difference recorded)" "$A/hf-cuda-two-way.json" "$A/official-two-way-1.json" 1 "$(c1_log 1)"
  elif selected C1; then
    C_ROWS+=("| C1 HF cuda two-way vs h2o TWO_WAY=1 (answers; difference recorded) | - | - | SKIP ($TWO_WAY_SKIP) |")
  fi
  compare C1 "HF cuda one-way vs h2o TWO_WAY=0 (answers; difference recorded)" "$A/hf-cuda-one-way.json" "$A/official-two-way-0.json" 1 "$(c1_log 0)"
  G="$A/hf-cuda-one-way-gguf-source"
  compare_gguf C2 "GGUF Q8_0 vs HF cuda one-way at the GGUF's source revision, text" "$A/gguf-Q8_0-text.json" "$G-text.json" 0.05 logs/llama-server-Q8_0.log
  compare_gguf C3 "GGUF Q4_K_M vs HF cuda one-way at the GGUF's source revision, text" "$A/gguf-Q4_K_M-text.json" "$G-text.json" 0.10 logs/llama-server-Q4_K_M.log
  compare_gguf C4 "GGUF Q8_0 vs HF cuda one-way at the GGUF's source revision, images (answers; difference recorded)" "$A/gguf-Q8_0-image.json" "$G-image.json" 1 logs/llama-server-Q8_0.log
  compare_gguf C4 "GGUF Q4_K_M vs HF cuda one-way at the GGUF's source revision, images (answers; difference recorded)" "$A/gguf-Q4_K_M-image.json" "$G-image.json" 1 logs/llama-server-Q4_K_M.log
  # For the record only (no pass mark): how far the GGUF build is from the HF default, v1.3.1,
  # the answers a user of the GGUF gets beside one of the HF build at its default revision.
  for quant in Q8_0 Q4_K_M; do
    record "C-gguf-$quant-vs-v1.3.1" "GGUF $quant vs HF cuda one-way v1.3.1, text and images" "$A/gguf-$quant.json" "$A/hf-cuda-one-way.json"
  done
  ;;
cpu)
  row U2a-python "docs examples text and image (cpu)" "$PY" "$HERE/docs_examples.py" --docs "$DOCS" --only text,image
  row U2a-cli "ask two-way, text and image (cpu)" u_cli_readings cpu two-way
  row U2a-serve "serve --vision, image request (cpu)" u_serve 1 "$HF" "${REV[@]}" --device cpu --vision
  gguf_rows U2b cpu T8 Q4_K_M Q8_0
  row T7 "time HF build, two-way and one-way, text and image (cpu)" bench_hf T7 cpu two-way one-way
  ;;
mac)
  row U3a-python "docs examples text and image (mps)" "$PY" "$HERE/docs_examples.py" --docs "$DOCS" --only text,image
  row U3a-cli "ask two-way, text and image (mps)" u_cli_readings mps two-way
  row U3a-serve "serve --vision, image request (mps)" u_serve 1 "$HF" "${REV[@]}" --device mps --vision
  row U3b-python "docs example light-mlx" "$PY" "$HERE/docs_examples.py" --docs "$DOCS" --only light-mlx
  row U3b-cli "ask MLX q8 (mlx)" u_cli "$MLX_Q8_REPO" --revision "$MLX_Q8_REV" --device mlx
  row U3b-serve "serve MLX q8 (mlx), text request" u_serve 0 "$MLX_Q8_REPO" --revision "$MLX_Q8_REV" --device mlx
  row C-mlx-q8 "MLX q8 answers, text" parity mlx-q8 "$MLX_Q8_REPO" --revision "$MLX_Q8_REV" --device mlx --text-only
  row C-mlx-q4 "MLX q4 answers, text" parity mlx-q4 "$MLX_Q4_REPO" --revision "$MLX_Q4_REV" --device mlx --text-only
  row T4-q8 "time MLX q8 (mlx), text" bench T4-q8 "$MLX_Q8_REPO" --revision "$MLX_Q8_REV" --device mlx
  row T4-q4 "time MLX q4 (mlx), text" bench T4-q4 "$MLX_Q4_REPO" --revision "$MLX_Q4_REV" --device mlx
  row T5 "time HF build, two-way and one-way, text and image (mps)" bench_hf T5 mps two-way one-way
  row T6 "time strands-decider-2B (text, image) and E4B (text) (mps)" bench_deciders T6 mps
  REF="${REFERENCE:+$REFERENCE/reference/hf-cuda-one-way.json}"
  if [ -n "$REF" ] && [ -f "$REF" ]; then
    cp "$REF" "$A/hf-cuda-one-way.json" && split_answers "$A/hf-cuda-one-way.json"
    compare C5 "MLX q8 vs HF cuda one-way, text" "$A/mlx-q8.json" "$A/hf-cuda-one-way-text.json" 0.05
    compare C6 "MLX q4 vs HF cuda one-way, text" "$A/mlx-q4.json" "$A/hf-cuda-one-way-text.json" 0.10
  else
    C_ROWS+=("| C5, C6 | - | - | FAIL (give --reference DIR, a gpu run's --out) |")
    FAILED=1
  fi
  ;;
esac

# ---------------------------------------------------------------- tables
say "## U: use cases, and the runs behind C and T"
say ""
say "| row | what | result | time | log |"
say "|---|---|---|---|---|"
for r in "${ROWS[@]}"; do say "$r"; done
say ""
if [ "${#C_ROWS[@]}" -gt 0 ]; then
  say "## C: answers compared"
  say ""
  say "| row | agree | largest probability difference | result |"
  say "|---|---|---|---|"
  for r in "${C_ROWS[@]}"; do say "$r"; done
  say ""
fi
breakdown_section
if [ -n "$ONLY" ]; then
  say "## T: not run (--only $ONLY)"
else
  say "## T: time per decision (bench_local.py, median and p90 of $REPS)"
  say ""
  t_rows | tee -a "$RESULTS"
fi
say ""
stopped_as="the GGUF rows were STOPPED (the C-hf-gguf-source row gives the reason)"
if [ "$FAILED" = 0 ]; then
  done_as="every row passed"
  [ "$SKIPPED" = 1 ] && done_as="every row passed or was skipped (SKIP rows give the reason)"
  [ "$STOPPED" = 1 ] && done_as="$done_as; $stopped_as"
  say "finished $(date -u +%Y-%m-%dT%H:%M:%SZ): $done_as"
else
  done_as="some rows FAILED"
  [ "$STOPPED" = 1 ] && done_as="$done_as; $stopped_as"
  say "finished $(date -u +%Y-%m-%dT%H:%M:%SZ): $done_as"
fi
exit "$FAILED"
