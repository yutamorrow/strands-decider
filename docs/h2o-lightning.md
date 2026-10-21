# h2o-lightning-4b

`strands-decider ask` and `strands-decider serve` also read
[`h2oai/h2o-lightning-4b`](https://huggingface.co/h2oai/h2o-lightning-4b), a decision model from
H2O.ai built on Qwen3.5-4B. It is served through the same `POST /v1/systemone` API, and its
answers have the same shape as a strands-decider checkpoint's. No h2o code is run: this package
reads the model's weights and its two serving files and does the readout itself.

## Using it

```bash
strands-decider ask h2oai/h2o-lightning-4b \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"

strands-decider serve h2oai/h2o-lightning-4b --port 8000
```

A checkpoint is read as h2o-lightning when it has `serve_config.json` and has neither
`strands_decider_config.json` nor `hobson_config.json`. The same rule applies to a local folder
and to a Hub repo. Everything else loads exactly as before. For a Hub id, the check fetches
`serve_config.json` once before anything else is downloaded, so loading a strands-decider repo
by its Hub id makes one extra request (which finds no such file); a local folder makes no
network request for the check.

Options that apply to h2o-lightning checkpoints:

- `--attention two-way` (the default) or `--attention one-way`. The weights were trained to answer
  under both. Two-way makes the full-attention layers attend in both directions; the Gated
  DeltaNet layers stay causal. One-way is the ordinary causal forward. `--attention` is not used
  for strands-decider checkpoints. From Python: `load_engine(..., attention="one-way")` and
  `create_app(..., attention="one-way")`.
- `--revision REV` downloads that branch, tag or commit of the Hub repo. Without it, the repo's
  default branch (`main`) is used. `main` changes as h2o publishes updates, so pin a revision
  for results you want to repeat. `--revision` works for strands-decider checkpoints too.

Only these files are downloaded from the repo (about 9.3 GB, almost all of it
`model.safetensors`); a name the repo does not have is skipped:

```
config.json  model.safetensors  tokenizer.json  tokenizer_config.json  chat_template.jinja
special_tokens_map.json  vocab.json  merges.txt  serve_config.json  label_rows.lightning
```

The repo's shim, adapter, scripts and assets are not downloaded, and `trust_remote_code` is never
set.

Limits:

- Text only. A request with `images`, `ask --image` and `serve --vision` are refused (HTTP 422
  for a request, a non-zero exit for `ask`).
- A question may have at most as many options as `serve_config.json` has labels; more is an HTTP
  422 (a non-zero exit for `ask`).
- A state longer than `serve_config.json`'s `max_state_tokens`, or a prompt longer than the
  model's context window, is refused rather than shortened.
- Each question is read as one whole prompt. In two-way reading the state's positions depend on
  the question that follows, so the state is not cached and shared between questions, and a
  request with many questions costs about that many forwards of the whole prompt.
- `--device mlx` (Apple silicon, the `mlx` extra) runs the same readout through mlx-lm. It is
  checked against torch on a tiny random model on a Mac and by hand on the real weights (below);
  CI on Linux does not run it. If MLX cannot load or run the model, the error says why and points
  to `--device cpu` (or `cuda`, `mps`).
- On CPU the 4B weights are read in fp32, about 17 GB of memory.

`GET /health` returns the same keys as for a strands-decider checkpoint: `base_model` is the one
`serve_config.json` names, `num_slots` is the number of labels, `max_length` is the context window,
and `temperature` is the per-type temperatures of the reading served, for example
`{"choice": 0.75, "noul": 0.8, "score": 0.65}`.

## How it reads an answer

Each question is rendered as `serve_config.json` fixes it: the chat template with thinking off,
the system prompt, then a user turn of `record: <state>`, `question: <instructions>`, `options:` and
one `<label>) <name>: <description>` line per option, and finally the `Answer:` prefill. A structured
state is written as compact JSON; a yes/no question's options are `true` and `false`, and a score's
options are its levels in order.

The prompt is tokenised once and forwarded once. The final hidden state at the last position
(after the final norm) is multiplied with the rows of the label tokens: in two-way reading the
rows in `label_rows.lightning`, in one-way reading the model's output embedding rows for the same
token ids. `label_rows.lightning` lists the token ids of each label; a label read from several
tokens is pooled with logsumexp. The labels' log-probabilities, normalised over the asked labels,
are divided by the question type's temperature and softmaxed. In two-way reading the temperatures
and the yes/no floor of `serve_config.json`'s `two_way` block are used, and the one-way values where
the block has none. A yes/no probability inside the floor moves to the floor on its own side.

Confidence follows this package's definitions (`derive_confidence` for choice,
`derive_score_confidence` for score), so it is comparable with strands-decider checkpoints, not
with the confidence h2o's own server reports for a score.

## Checking the numbers

The tests use a tiny random model, so they cannot show that the real weights give the answers
h2o's own serving gives. This check does, by hand. The pass criteria are fixed here, before any
result is seen, and are not changed afterwards.

**GPU.** On one NVIDIA GPU instance, serve the same revision of `h2oai/h2o-lightning-4b` four ways
and send each the 12 requests below:

1. h2o's serving, two-way: `TWO_WAY=1 bash serve.sh` (the repo's `serve.sh`, from the same
   revision). Two-way needs h2o's per-request-attention vLLM fork
   ([h2oai/vllm](https://github.com/h2oai/vllm)) installed as the model card describes; stock
   vLLM does not serve it.
2. h2o's serving, one-way: `TWO_WAY=0 bash serve.sh`, on vLLM 0.30.0 with h2o's shim.
3. `strands-decider serve h2oai/h2o-lightning-4b --revision REV --device cuda --attention two-way`
4. `strands-decider serve h2oai/h2o-lightning-4b --revision REV --device cuda --attention one-way`

Compare 1 with 3 and 2 with 4.

**Mac.** On an Apple-silicon Mac, serve `--device mlx` and `--device cpu` (torch) at the same
revision, in both readings, send the same 12 requests, and compare MLX with torch for each
reading. Every request must return an answer.

**Pass.** For every request, the answers agree (choice: the option chosen; score: the most likely
level; yes/no: the side of 0.5), and the largest absolute difference between the two servers'
probabilities, over every option of every question, is at most 0.02. The reason for 0.02: h2o
reports that its two readouts of the same attention differ by about 0.01 in bf16, and this
package computes in a different order and precision than vLLM, so twice that is allowed. Record
the revision, the GPU, the Mac model, the number of requests that agree and the largest
difference with the result.

The 12 requests, one JSON body per line (4 yes/no, 4 choice, 4 score):

```json
{"state": "Checkout returns HTTP 500 for every customer since 09:12.", "questions": {"q": {"type": "noul", "instructions": "Is this a full outage?"}}}
{"state": "The parcel left the depot on Monday, was scanned in Lyon on Tuesday and has not moved since.", "questions": {"q": {"type": "noul", "instructions": "Should the parcel be declared lost?"}}}
{"state": {"from": "dana@example.com", "subject": "Invoice 4471", "body": "Please find the overdue invoice attached."}, "questions": {"q": {"type": "noul", "instructions": "Does this email ask the reader to pay something?"}}}
{"state": "def add(a, b):\n    return a - b", "questions": {"q": {"type": "noul", "instructions": "Does the function do what its name says?", "criteria": {"true": "it adds its arguments", "false": "it does something else"}}}}
{"state": "Help! My payouts have been failing for 3 days!", "questions": {"q": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "payments and payouts", "sales": "new contracts", "retail": "store orders"}}}}
{"state": "sihamba ngokushesha", "questions": {"q": {"type": "choice", "instructions": "What language is this phrase in?", "criteria": {"english": "English", "zulu": "Zulu", "dutch": "Dutch"}}}}
{"state": {"tool": "get_weather", "arguments": {"city": ""}}, "questions": {"q": {"type": "choice", "instructions": "What should the agent do with this tool call?", "criteria": {"run": "run it as is", "ask": "ask the user for the missing city", "drop": "drop the call"}}}}
{"state": "Battery lasts two days, screen is dim outdoors, support replaced it once without fuss.", "questions": {"q": {"type": "choice", "instructions": "What is the main complaint?", "criteria": {"battery": "Battery", "screen": "Screen", "support": "Support", "none": "No complaint"}}}}
{"state": "this is the best doc I've ever read", "questions": {"q": {"type": "score", "instructions": "How positive is this?", "criteria": ["negative", "neutral", "positive"]}}}
{"state": "Help! My payouts have been failing for 3 days!", "questions": {"q": {"type": "score", "instructions": "How frustrated is the writer?", "criteria": ["calm", "frustrated", "depressed"]}}}
{"state": "The server is at 92% disk and rising 1% an hour.", "questions": {"q": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "today", "within the hour", "now"]}}}
{"state": "The answer cites two sources; one of them does not mention the claim.", "questions": {"q": {"type": "score", "instructions": "How well supported is the answer?", "criteria": ["unsupported", "partly supported", "fully supported"]}}}
```

A script that sends them to two servers and reports the result (save the lines above as
`requests.jsonl`):

```python
import json, sys, urllib.request


def ask(url, body):
    req = urllib.request.Request(
        url + "/v1/systemone", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req))["answers"]["q"]


def dist(a):
    return (
        {"true": a["noul"], "false": 1 - a["noul"]} if a["type"] == "noul" else a["probabilities"]
    )


def pick(a):
    d = dist(a)
    return max(d, key=d.get)


a_url, b_url = sys.argv[1], sys.argv[2]
agree, worst = 0, 0.0
for line in open("requests.jsonl"):
    body = json.loads(line)
    a, b = ask(a_url, body), ask(b_url, body)
    agree += pick(a) == pick(b)
    worst = max(worst, *(abs(dist(a)[k] - dist(b)[k]) for k in dist(a)))
print(f"answers agree on {agree} of 12; largest probability difference {worst:.4f}")
print("PASS" if agree == 12 and worst <= 0.02 else "FAIL")
```

```bash
python compare.py http://127.0.0.1:8741 http://127.0.0.1:8000
```
