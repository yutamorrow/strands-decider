# Checking h2o-lightning-4b on a machine

`run_checks.sh` checks [h2o-lightning-4b](../../docs/h2o-lightning.md) through strands-decider on
one machine, installed the way a user installs it, and writes the result tables to
`DIR/results.md` (and to the terminal). The tests in `tests/` use tiny random models; this is the
check on the real weights.

## Running it

Install strands-decider from the commit to check, in a new virtual environment, and fetch this
directory and `docs/h2o-lightning.md` from the same commit (or run from a clone of it):

```bash
pip install "strands-decider[vision] @ git+https://github.com/yutamorrow/strands-decider@SHA"
# on an Apple-silicon Mac: "strands-decider[vision,mlx] @ git+...@SHA"

evaluation/h2o_lightning/run_checks.sh --profile gpu --out gpu-checks   # Linux with an NVIDIA GPU
evaluation/h2o_lightning/run_checks.sh --profile cpu --out cpu-checks   # Linux without a GPU
evaluation/h2o_lightning/run_checks.sh --profile mac --out mac-checks --reference gpu-checks
```

To run only the GGUF rows again on a GPU machine, add `--only gguf` to the `gpu` profile:

```bash
evaluation/h2o_lightning/run_checks.sh --profile gpu --only gguf --out gpu-gguf-checks
```

It runs `C-hf-gguf-source`, `C-hf` (the v1.3.1 one-way answers only, the record-only rows'
reference; not two-way), `C-gguf-Q8_0` and `C-gguf-Q4_K_M`, then compares C2-C4 and the
record-only rows; no U, C1 or T row, so nothing is timed. The header's `only` row says `gguf`
(`all rows` without `--only`). Another group, or `--only gguf` with another profile, is refused
with the usage and exit code 2.

`--dry-run` writes only the header. The header gives the strands-decider commit (`git rev-parse
HEAD` in a clone, else the `commit_id` pip recorded), the Python package versions, the machine and
every pinned revision. The revisions are fixed at the top of the script: h2o-lightning-4b
`e5c86b7564de9574fc48e759c226e50819d079ec` (v1.3.1), the MLX builds made from it, the GGUF build
(tag v1.2.3, made from the earlier revision `9030a28`; see below), llama.cpp `b11854`, vLLM
0.30.0 for h2o's own serving, and the strands-decider models timed beside it.

Nothing needs credentials: every model is a public Hugging Face repo fetched anonymously. Every
server the script starts listens on 127.0.0.1 only, and is stopped when the script ends.

C1 runs h2o's own scripts from the pinned h2o-lightning-4b revision (`serve.sh` and its shim), and
they listen on 127.0.0.1: at that revision `serve.sh` starts vLLM with `--host 127.0.0.1` and the
shim on `$SHIM_HOST`, which the script sets to `127.0.0.1`. The script also empties `VLLM_ARGS`,
so no inherited `vllm serve` flag can change the address.

C1 uses stock vLLM 0.30.0 in its own virtual environment, as the pinned revision's model card
does for these two layouts: `TWO_WAY=1` is one pooling-runner engine and `TWO_WAY=0` one
generate-mode engine; neither needs h2o's vLLM fork or a patch (the fork is only for serve.sh's
`ONE_ENGINE=1`, which the script does not run). Both run with serve.sh's own settings
(`MAX_LEN` 40960, `GPU_UTIL` 0.90, `MAX_IMAGES` 4, `MAX_PIXELS` from `serve_config.json`), and
`VLLM_ARGS` stays empty; `OFFICIAL_TWO_WAY_MAX_LEN`, when set, is passed to `TWO_WAY=1` as
`MAX_LEN` and changes nothing else.

**C1 two-way needs a GPU of 48 GB or more** (an L40S, as on a g6e instance). The two-way engine
runs with `--no-enable-chunked-prefill`, so vLLM 0.30.0 (`engine/arg_utils.py`,
`config/scheduler.py`, `multimodal/encoder_budget.py`, `profile_run` in
`v1/worker/gpu_model_runner.py`) sets `max_num_batched_tokens` and the encoder cache budget to
`max(MAX_LEN, 2048)` and, at start-up, profiles the vision tower on that many image tokens and
one forward of that many tokens, with no memory cap. What sets the memory is `MAX_LEN` (the size
of that profile) and `GPU_UTIL` (vLLM's share of the GPU). `MAX_IMAGES` is only the limit on
images per request (`--limit-mm-per-prompt`): the profile's image count comes from the encoder
budget, so it does not change the profile's size. On a 24 GB L4 the engine did not start at the
shipped `MAX_LEN` 40960, nor at 8192 ("Tried to allocate 256.00 MiB", 21.89 GiB in use), nor at
2048 with `GPU_UTIL` 0.80. So on a GPU that reports less than `OFFICIAL_TWO_WAY_MIN_MIB` (44000
MiB; a 48 GB card reports less than 48 GiB, an L40S 46068 MiB) the `gpu` profile skips C1
two-way: the `C1-official-1` row and the C1 two-way comparison say `SKIP` with the reason ("C1
two-way needs a GPU of 48 GB or more; this GPU has 22 GiB (23034 MiB)"), and a skip is not a
failure. C1 one-way (`TWO_WAY=0`) runs on any GPU. When h2o's serving fails, the script keeps
the first error of the server log in `logs/official-two-way-<n>-first-error.log` (and in the
row's log), and the C1 row of the C table points at that file: the first `OutOfMemory` line, else
the first `Error` line, and, when that line ends a traceback, the whole of that traceback from
vLLM's "EngineCore failed to start." on, so the failing step's frame is in it (an unrelated
traceback earlier in the log is not taken in).

The GGUF build: **the official GGUF (`2504cec`, the GGUF repo's tag v1.2.3) is made from the
v1.2.3 weights, a different release from the HF build's default (v1.3.1).** Its files are
quantised from h2o-lightning-4b at `9030a28dac81fcffd328f927e41a5deb85a27e5f` (the GGUF repo's
`artifact_manifest_v3.json`, `source_model.revision`), whose weights (`model.safetensors` sha256
`2d51561f221e...`, as at h2o's tags v1.2 to v1.2.3) are not v1.3.1's (`0779980b4335...`); the
GGUF repo has no files of v1.3.x. To measure quantisation the weights have to be the same, so
C2-C4 compare the GGUF answers with the HF build at that revision (`GGUF_SOURCE_REV`), read
one-way, and the GGUF rows of the C table take their prompt settings from it too (its
`serve_config.json` one-way settings, `tokenizer.json` and `chat_template.jinja` are v1.3.1's).
The pass marks are unchanged. The `gpu` profile's first row, `C-hf-gguf-source`, checks that
this strands-decider reads that revision (torch, one-way; it has no `label_rows.lightning`,
which only two-way needs) and answers the requests. When it cannot, the row says `STOPPED` with
the reason ("the HF build at the GGUF's source revision 9030a28 could not be read: <the error
line>; logs/C-hf-gguf-source.log"), and so does every row that depends on that revision, with
the same reason: `C-gguf-Q8_0` and `C-gguf-Q4_K_M` (the GGUF answers, whose prompt settings come
from it; they are not run) and C2-C4. This holds whichever side of the revision could not be
read, its weights or its settings files. A stop is not a failure: those rows are not counted as
failed, and the last line says the GGUF rows were STOPPED (and also that some rows FAILED, when
other rows did). There is then no reference with the GGUF's own weights. When
`C-hf-gguf-source` passes, a `C-gguf-<quant>` row that fails (the llama-server's side) is a
FAIL, as are C2-C4 without its answers. Compared with v1.3.1 instead, as the fourth GPU run did,
the two builds agree on every text answer but their probabilities differ by up to 0.20 at Q8_0
and 0.19 at Q4_K_M, about the same at both precisions; the weights differ, and the other inputs
of the reading (tokenizer, chat template, image processor, one-way settings) are the same, so the
difference is taken to come mainly from the weights. Against the GGUF's own source revision (the
GPU run on an L40S at `5a0b79b`), C2 (Q8_0) agreed on 12 of 12 text requests with a largest
difference of 0.0163, within its 0.05; C3 (Q4_K_M) agreed on 11 of 12 with 0.6020, over its 0.10,
and stays a FAIL (the limit is not changed); C4 agreed on 4 of 4 images at both precisions (0.0013
and 0.0679). The one C3 request that changed is `noul-invoice` (a yes/no question): the HF build
gives yes 0.8010, Q4_K_M gives no (yes 0.1990), and Q8_0 matches the HF build exactly. 0.801 is
h2o's `noul_floor`, so both readings were under it on opposite sides of 0.5 and the 0.6020 is the
gap between the two floored values; the probability read is by token id with the same bias on
both labels, as for every other request, so this is a close answer that the 4-bit weights tipped.
The other 11 text requests are within 0.0244. Hence the docs' advice: Q4_K_M can change an
answer; if accuracy matters, use Q8_0. That difference stays in the
C table as rows for the record only, `C-gguf-Q8_0-vs-v1.3.1` and `C-gguf-Q4_K_M-vs-v1.3.1` (the
answers that agree and the largest difference against `hf-cuda-one-way.json`, v1.3.1), whose
result is "record only": they pass or fail nothing. When the two answer files cannot be compared
(one is missing, or `h2o_parity.py` stops or raises), the row keeps its ID and says "record only:
could not compare (<the reason in one line>)"; `h2o_parity.py`'s error output is in
`logs/C-gguf-<quant>-vs-v1.3.1.log`, which the row names.

The GGUF rows need a llama-server: set `LLAMA_SERVER_BIN` to a `llama-server` binary of build
b11854 or later (the CUDA build on a GPU machine), or have docker, and the script runs the pinned
llama.cpp container (`server-cuda-b11854` on `gpu`, `server-b11854` on `cpu`). With neither,
those rows fail; they are never skipped. The script starts it with the vision projector, as
`docs/h2o-lightning.md` does. The GGUF and the projector come from the Hugging Face cache, whose
snapshot files are symlinks into `blobs/`; the script resolves each with `realpath` and mounts
those two files read-only into the container (a mount of the snapshot directory alone leaves the
links dangling there). The container runs without `--rm`, so its output survives it: when the
container, or the `LLAMA_SERVER_BIN` process, stops before `/health` answers, the row fails at
once, and the server's output is in `logs/llama-server-<quant>.log` (from `docker logs` for the
container), which the failed row names. The container is removed when its rows are done. Images reach the server at the size and with the
pixels the HF build's image processor reads (`src/strands_decider/h2o_llama.py` explains why);
the script starts it with `--image-max-tokens 1024`, h2o's setting, so an image whose HF size
rounds past 1024 tokens (only one near `image.max_pixels`) is shrunk again by the server. C4's
drawn images are 140 tokens.

Why b11854: the llama-server API strands-decider uses was read in llama.cpp's source at b11856,
which has no container on ghcr.io. The nearest builds with both `server-b<N>` and
`server-cuda-b<N>` are b11854 and b11870. From b11854 to b11856 only CI and tests changed
(`ci/`, `.github/`, `tests/`); `tools/server/` and `tools/mtmd/` are the same git trees, so
`/completion` (`n_probs`, `logit_bias`, `post_sampling_probs`, `completion_probabilities[0].top_probs`
with `id` and `prob`), `GET /props` `media_marker` and the `{prompt_string, multimodal_data}`
prompt are as read. b11870 rewrites the server entrypoint (llama.cpp #31113) and changes the
`tools/mtmd` image code, so it was not chosen.

Other settings, as environment variables: `PYTHON` (default `python3`), `PORT` (strands-decider
serve, 8790), `LLAMA_PORT` (8080), `OFFICIAL_PORT` (h2o's serving, 8741),
`OFFICIAL_TWO_WAY_MIN_MIB` (the GPU memory C1 two-way needs, 44000 MiB) and
`OFFICIAL_TWO_WAY_MAX_LEN` (serve.sh's `MAX_LEN` for `TWO_WAY=1`, unset: serve.sh's 40960),
`LENGTHS` (state
tokens timed, `256,1024,4000`), `QUESTIONS` (`1,4`) and `REPS` (5).

## What each profile runs

| Profile | Rows | Who runs it |
|---|---|---|
| `gpu` | U1a, U1b, C1-C4, T1-T3 (C1 two-way on a GPU of 48 GB or more) | the agent, on a GPU machine it starts |
| `cpu` | U2a, U2b, T7, T8 | the agent, on a CPU machine it starts |
| `mac` | U3a, U3b, C5, C6, T4-T6 | the person, on their Mac |

**U: use cases.** On each machine, the docs' Python examples (`docs_examples.py`, which runs the
marked blocks of `docs/h2o-lightning.md` exactly as written), `strands-decider ask` and
`strands-decider serve` with a `POST /v1/systemone` (with an image where the row says so). A row
passes when every example ends without an exception, `ask` exits 0, `serve` answers 200, and
every answer validates as a `SystemOneResponse`.

| Row | Machine | Build | Reading | Input |
|---|---|---|---|---|
| U1a | GPU Linux (cuda) | HF | two-way, one-way | text, image |
| U1b | GPU Linux | GGUF Q8_0 (llama-server, CUDA build) | one-way | text, image |
| U2a | CPU Linux (cpu) | HF | two-way | text, image |
| U2b | CPU Linux | GGUF Q4_K_M (llama-server, CPU build) | one-way | text, image |
| U3a | Mac (mps) | HF | two-way | text, image |
| U3b | Mac (mlx) | MLX q8 | one-way | text |

**C: answers compared.** `../h2o_parity.py` sends the 12 text requests of `docs/h2o-lightning.md`
and 4 requests over images it draws, and compares two answer files. Two answers agree when they
choose the same option (yes/no: the side of 0.5; choice: the most likely option; score: the most
likely level). The criteria were fixed before any result was seen and are not changed afterwards.

| Row | Compared | Against | Passes when |
|---|---|---|---|
| C1 | HF on cuda, two-way and one-way, text and images | h2o's own serving (stock vLLM 0.30.0 with `serve.sh` as shipped, `TWO_WAY=1` on a GPU of 48 GB or more, skipped on a smaller one, and `TWO_WAY=0`) | every answer agrees (the largest difference is recorded) |
| C2 | GGUF Q8_0, text | HF on cuda, one-way, text, at the GGUF's source revision (`9030a28`) | every answer agrees and the largest difference is at most 0.05 |
| C3 | GGUF Q4_K_M, text | the same | every answer agrees and the largest difference is at most 0.10 |
| C4 | GGUF Q8_0 and Q4_K_M, images | HF on cuda, one-way, images, at the GGUF's source revision | every answer agrees (the largest difference is recorded) |
| C-gguf-*-vs-v1.3.1 | GGUF Q8_0 and Q4_K_M, text and images | HF on cuda, one-way, v1.3.1 | record only (no pass mark) |
| C5 | MLX q8, text | HF on cuda, one-way, text (the gpu run's file) | every answer agrees and the largest difference is at most 0.05 |
| C6 | MLX q4, text | the same | every answer agrees and the largest difference is at most 0.10 |

The light builds are compared with the HF build's one-way reading because they read one-way only,
each at the revision it was made from: the MLX builds with v1.3.1, the GGUF build with
`9030a28`. The `gpu` profile writes the v1.3.1 reference to `DIR/reference/hf-cuda-one-way.json`;
give the gpu run's `DIR` to the `mac` profile with `--reference` (copy the directory to the Mac).
The criteria are unchanged; only the GGUF rows' reference moved to the revision their weights
are from, and when this strands-decider cannot read that revision those rows (C2-C4 and the
GGUF answers they compare) are `STOPPED`, not failed (see above).

C2-C4 and the record-only rows also show their comparison request by request, in a section of
`results.md` after the C table (one table per row, `h2o_parity.py --details`, kept in
`DIR/details/`): the request, the reference's answer and the compared one, whether they agree,
the largest probability difference on that request and, where the answers differ, the three most
likely options of each with their probabilities. Each request is also printed on the terminal
(stderr) as `CHECK-DIFF <row> <answers file> | request | ... |`, for example `CHECK-DIFF C3
gguf-Q4_K_M-text | ...`. A row that could not compare (`STOPPED`, a missing answer file, "could
not compare") has no such table. This shows the answers only: the pass marks are computed as
before.

**T: time per decision.** `../bench_local.py` times requests at the state lengths in `LENGTHS`
and the question counts in `QUESTIONS`, `REPS` times each, and reports the median and p90. Time
has no pass mark: it shows which build to pick on which machine.

| Row | Build | Reading | Input | Machine |
|---|---|---|---|---|
| T1 | HF | two-way, one-way | text, image | GPU Linux (cuda) |
| T2 | GGUF Q8_0, Q4_K_M (CUDA build) | one-way | text, image | GPU Linux |
| T3 | strands-decider-2B; E4B | as shipped | 2B text and image; E4B text | GPU Linux (cuda) |
| T4 | MLX q8, q4 | one-way | text | Mac (mlx) |
| T5 | HF | two-way, one-way | text, image | Mac (mps) |
| T6 | strands-decider-2B; E4B | as shipped | 2B text and image; E4B text | Mac (mps) |
| T7 | HF | two-way, one-way | text, image | CPU Linux (cpu) |
| T8 | GGUF Q8_0, Q4_K_M (CPU build) | one-way | text, image | CPU Linux |

strands-decider-E4B is a Gemma 4 model, which reads text only (docs/vision.md), so it is timed on
text only. On CPU the HF build is read in fp32 (about 17 GB of weights), so U2a and T7 need a
machine with about 32 GB of memory; U2a passes when it answers, however slowly.

## Output

`DIR/results.md` holds the header and the U, C and T tables; `DIR/logs/` one log per row (and
the server logs: `llama-server-<quant>.log`, `official-two-way-<n>-server.log` and, when h2o's
serving failed, `official-two-way-<n>-first-error.log`),
`DIR/answers/` the answer files, `DIR/details/` the comparisons per request, and `DIR/bench/`
the timing CSVs. The script runs every row even
when an earlier one fails, and exits 1 when any row failed.
