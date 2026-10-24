# h2o-lightning-4b

strands-decider also reads [`h2oai/h2o-lightning-4b`](https://huggingface.co/h2oai/h2o-lightning-4b),
a decision model from H2O.ai built on Qwen3.5-4B, from Python (`load_engine`), from `strands-decider
ask` and from `strands-decider serve`. It is served through the same `POST /v1/systemone` API, and
its answers have the same shape as a strands-decider checkpoint's. No h2o code is run: this
package reads the model's weights and its serving files and does the readout itself.

Three builds of the model can be used:

| Build | Where it runs | Readings | Images |
|---|---|---|---|
| HF: `h2oai/h2o-lightning-4b` | torch: `cuda`, `mps`, `cpu`; or `--device mlx` | two-way (default), one-way | yes, on torch |
| MLX-quantised: for example `netprojects/h2o-lightning-4b-mlx-q8`, `-q4` | `--device mlx` (Apple silicon) | one-way | no |
| GGUF: `h2oai/h2o-lightning-4b-gguf` | a llama.cpp `llama-server` you start (`--engine llama-server`) | one-way | yes, with the vision projector |

## Installing

```bash
pip install "strands-decider[vision]"        # Linux, with or without a GPU
pip install "strands-decider[vision,mlx]"    # Apple-silicon Mac
```

The `vision` extra (Pillow) is needed for images; `mlx` for `--device mlx`. Images on torch
devices (`cuda`, `mps`, `cpu`) also need transformers 5.18 or later, newer than the 5.15 that
text needs (`pip install "transformers>=5.18"`).

## Using it from Python

`load_engine` gives an engine whose `evaluate` takes a `SystemOneRequest` and returns a
`SystemOneResponse`. Pin a revision for results you want to repeat; without one, the repo's
default branch is used, and it changes as h2o publishes updates.

<!-- example: text -->
```python
import torch

from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
engine = load_engine(
    "h2oai/h2o-lightning-4b",
    revision="e5c86b7564de9574fc48e759c226e50819d079ec",  # v1.3.1
    device=device,
)
response = engine.evaluate(
    SystemOneRequest(
        state="Help! My payouts have been failing for 3 days!",
        questions={
            "team": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "payments and payouts", "sales": "new contracts"},
            ),
            "urgent": NoulQuestion(instructions="Does this convey urgency?"),
            "mood": ScoreQuestion(
                instructions="How frustrated is the writer?",
                criteria=["calm", "frustrated", "angry"],
            ),
        },
    )
)
print(response.model_dump_json(indent=1))
```

With `vision=True` the engine also loads the vision tower, and a request may carry images in
`images`, base64-encoded (PNG, JPEG, WEBP or GIF; here one is drawn with Pillow):

<!-- example: image -->
```python
import base64
import io

import torch
from PIL import Image, ImageDraw

from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion, SystemOneRequest

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
engine = load_engine(
    "h2oai/h2o-lightning-4b",
    revision="e5c86b7564de9574fc48e759c226e50819d079ec",
    device=device,
    vision=True,
)
picture = Image.new("RGB", (448, 336), "white")
ImageDraw.Draw(picture).ellipse((112, 84, 336, 252), fill="red")
buffer = io.BytesIO()
picture.save(buffer, format="PNG")  # or the bytes of an image file
response = engine.evaluate(
    SystemOneRequest(
        state="A picture from the shape sorter.",
        questions={
            "shape": ChoiceQuestion(
                instructions="Which shape is shown?",
                criteria={"circle": "a circle", "square": "a square", "triangle": "a triangle"},
            )
        },
        images=[base64.b64encode(buffer.getvalue()).decode("ascii")],
    )
)
print(response.answers["shape"].choice)
```

The MLX-quantised build, on an Apple-silicon Mac:

<!-- example: light-mlx -->
```python
from strands_decider.infer import load_engine
from strands_decider.schema import NoulQuestion, SystemOneRequest

engine = load_engine(
    "netprojects/h2o-lightning-4b-mlx-q8",
    revision="12a62e9c21e5630fbe16da1e52377d025800fe8e",
    device="mlx",
)  # one-way, the light builds' default
response = engine.evaluate(
    SystemOneRequest(
        state="Checkout returns HTTP 500 for every customer since 09:12.",
        questions={"outage": NoulQuestion(instructions="Is this a full outage?")},
    )
)
print(response.answers["outage"].noul)
```

The GGUF build, through a llama-server already running on this machine (see
[The GGUF build through llama-server](#the-gguf-build-through-llama-server)). The h2o-lightning
repo gives only the prompt settings and the tokenizer; its weights are not downloaded:

<!-- example: light-llama -->
```python
from strands_decider.infer import load_engine
from strands_decider.schema import NoulQuestion, SystemOneRequest

engine = load_engine(
    "h2oai/h2o-lightning-4b",
    revision="e5c86b7564de9574fc48e759c226e50819d079ec",
    engine="llama-server",  # at $STRANDS_DECIDER_LLAMA_SERVER_URL, else http://127.0.0.1:8080
)
response = engine.evaluate(
    SystemOneRequest(
        state="Checkout returns HTTP 500 for every customer since 09:12.",
        questions={"outage": NoulQuestion(instructions="Is this a full outage?")},
    )
)
print(response.answers["outage"].noul)
```

`server.create_app` takes the same arguments (`attention`, `revision`, `vision`, `engine`,
`llama_server_url`) and returns the FastAPI app `serve` runs.

## From the command line

```bash
strands-decider ask h2oai/h2o-lightning-4b --revision e5c86b7564de9574fc48e759c226e50819d079ec \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"

strands-decider ask h2oai/h2o-lightning-4b --state "" --image page.png --noul "Is the form signed?"

strands-decider serve h2oai/h2o-lightning-4b --port 8000
strands-decider serve h2oai/h2o-lightning-4b --vision --port 8000   # requests may carry images
```

A checkpoint is read as h2o-lightning when it has `serve_config.json` and has neither
`strands_decider_config.json` nor `hobson_config.json`. The same rule applies to a local folder
and to a Hub repo. Everything else loads exactly as before. For a Hub id, the check fetches
`serve_config.json` once before anything else is downloaded, so loading a strands-decider repo
by its Hub id makes one extra request (which finds no such file); a local folder makes no
network request for the check.

Options that apply to h2o-lightning checkpoints:

- `--attention two-way` or `--attention one-way`. Without it, the HF build reads two-way and the
  light builds (MLX-quantised, GGUF) one-way. The weights were trained to answer under both.
  Two-way makes the full-attention layers attend in both directions, over the image tokens too;
  the Gated DeltaNet layers stay causal. One-way is the ordinary causal forward. The light builds
  read one-way only: asking them for two-way is an error. `--attention` is not used for
  strands-decider checkpoints. From Python: `load_engine(..., attention="one-way")`.
- `--revision REV` downloads that branch, tag or commit of the Hub repo. Without it, the repo's
  default branch (`main`) is used. `--revision` works for strands-decider checkpoints too.
- `--engine llama-server`, `--llama-server-url URL` and `--llama-server-timeout SECONDS`: see
  [The GGUF build through llama-server](#the-gguf-build-through-llama-server).

Only these files are downloaded from the HF repo (about 9.3 GB, almost all of it
`model.safetensors`); a name the repo does not have is skipped:

```
config.json  model.safetensors  tokenizer.json  tokenizer_config.json  chat_template.jinja
special_tokens_map.json  vocab.json  merges.txt  preprocessor_config.json  serve_config.json
label_rows.lightning
```

The repo's shim, adapter, scripts and assets are not downloaded, and `trust_remote_code` is never
set. With `--engine llama-server`, `model.safetensors` and `label_rows.lightning` are not
downloaded either.

## Images

`serve --vision`, `ask --image` and `load_engine(..., vision=True)` load the whole multimodal
model, vision tower included; without them only the text decoder is loaded, and a request with
`images` is refused (HTTP 422). Images are read on torch devices (`cuda`, `mps`, `cpu`) and by
the GGUF build; `--device mlx` refuses them before loading anything (HTTP 422 for a request, a
non-zero exit for `ask`, and `serve --vision --device mlx` does not start). On torch devices,
images need transformers 5.18 or later and the `vision` extra; with an older transformers,
loading for images stops with an error that says so. The GGUF build's images are read by the
llama-server and need only Pillow.

As in h2o's own serving, the images open the user turn, before `record:`; the rest of the
prompt is the text prompt. Each image is checked as `--vision` checks images (PNG, JPEG, WEBP or
GIF only, at most 4096 x 4096 pixels, EXIF rotation applied, transparency on white) and then
scaled down, keeping its aspect ratio, to at most `serve_config.json`'s `image.max_pixels`
(1,048,576 for v1.3.1). More images than `image.max_images` (4) is refused (HTTP 422). A request
with images uses the image temperature for every question type: `two_way.image_temperature` in
two-way, `image.temperature` in one-way (0.65 for both in v1.3.1). The vision tower runs once per
request, however many questions it has. A checkpoint whose `serve_config.json` has no `image`
block takes no images.

## MLX-quantised builds

A checkpoint whose `config.json` sets `quantization` (as mlx-lm writes it) is read as an
MLX-quantised build, with `--device mlx` only; a torch device refuses it before downloading the
weights. It is read as quantised, one-way by default. Without `label_rows.lightning` (the
community builds ship none) it reads one-way only, and each label is read from the single token
that follows `Answer:` in its tokenizer, as h2o builds its label rows (for v1.3.1 these are the
same token ids as in `label_rows.lightning`). MLX takes no images.

## The GGUF build through llama-server

strands-decider does not run the GGUF weights itself: you start a llama.cpp `llama-server`, and
`--engine llama-server` sends it each question's prompt. Use llama.cpp build b11773 or later (the
table files of the GGUF repo need it; checked with b11854), and the vision projector for images:

```bash
hf download h2oai/h2o-lightning-4b-gguf h2o-lightning-4b-Q8_0.gguf mmproj-lightning-f16.gguf \
  --revision 2504cec227e7df18d1764b6dff6ef37ed4666b1b --local-dir gguf
llama-server -m gguf/h2o-lightning-4b-Q8_0.gguf --mmproj gguf/mmproj-lightning-f16.gguf \
  -c 40960 -np 1 --no-kv-unified --image-max-tokens 1024 --host 127.0.0.1 --port 8080

strands-decider serve h2oai/h2o-lightning-4b --revision e5c86b7564de9574fc48e759c226e50819d079ec \
  --engine llama-server --port 8000
```

The checkpoint argument is the HF repo (or a local copy), read for its `serve_config.json` and
tokenizer, so the prompt, the labels, the temperatures and the yes/no floor are those of the HF
build. The prompt is tokenised here and sent as token ids to the server's `POST /completion` for
one token, with the same bias on every label token and the server's post-sampling probabilities
at temperature 1, so the probabilities that come back are the softmax over the label tokens (the
method of h2o's own llama.cpp backend). This is the one-way reading; two-way is refused. Images
go to the server as base64 PNG after the same checks and scaling as above, at the size and with
the pixels the HF build's image processor reads them at (its `preprocessor_config.json`
`smart_resize`, Pillow bicubic), in place of the server's media marker before `record:`, in a
prompt in Unicode NFC as the HF tokenizer normalises it. llama-server keeps an image that is
already at a valid size, so it reads the HF build's pixels, except for an image whose size rounds
past the server's `--image-max-tokens` (1024 above; only an image near `image.max_pixels`), which
the server shrinks again. llama.cpp treats image input for this model as experimental, and a
server started without `--mmproj` refuses images.

Which weights answer: **the official GGUF (`2504cec`, the GGUF repo's tag v1.2.3) is made from
the v1.2.3 weights, a different release from the HF build's default (v1.3.1).** Its files are
quantised from
`h2oai/h2o-lightning-4b` at `9030a28dac81fcffd328f927e41a5deb85a27e5f` (the GGUF repo's
`artifact_manifest_v3.json`), the weights of h2o's releases v1.2 to v1.2.3, not v1.3.1's; the
GGUF repo has no files of v1.3.x. Its answers therefore are not v1.3.1's: on the 16 requests of
the checks (12 text, 4 with an image), measured on an NVIDIA L40S, Q8_0 chose the same option as
the HF build at v1.3.1 on 15 and Q4_K_M on 14, with probabilities up to 0.77 apart. To get
answers that match an HF build, compare with the HF build at the revision
the GGUF was made from (`--revision 9030a28dac81fcffd328f927e41a5deb85a27e5f`, read one-way), as
`evaluation/h2o_lightning/run_checks.sh` does. The prompt settings of the two revisions are the
same (their one-way `serve_config.json` settings, tokenizer and chat template), so the
checkpoint argument above may be either.

Which file to use: **Q4_K_M can change an answer; if accuracy matters, use Q8_0.** Against the HF
build at the GGUF's own source revision (`9030a28`, one-way, the same weights), on an NVIDIA
L40S:

| file | text requests that agree | largest probability difference (text) | image requests that agree | largest difference (images) |
|---|---|---|---|---|
| Q8_0 | 12 of 12 | 0.0163 | 4 of 4 | 0.0013 |
| Q4_K_M | 11 of 12 | 0.6020 | 4 of 4 | 0.0679 |

The one Q4_K_M answer that changed is a yes/no question ("Does this email ask the reader to pay
something?" about an overdue invoice): the HF build answers yes at 0.801, Q4_K_M answers no at
0.801 (yes 0.199), and Q8_0 answers as the HF build does. 0.801 is h2o's yes/no floor
(`noul_floor` in `serve_config.json`): an answer less sure than that is reported at the floor on
its own side. So both builds were less sure than 0.801, on opposite sides of 0.5, and the 4-bit
weights tipped a close answer; the 0.6020 is the distance between the two floored values, not the
size of the shift underneath. On the other 11 text requests Q4_K_M is within 0.0244 of the HF
build.

Where the server is: `--llama-server-url URL`, else the environment variable
`STRANDS_DECIDER_LLAMA_SERVER_URL`, else `http://127.0.0.1:8080`. It is fixed when `ask` or
`serve` starts; nothing in a request can change it. An address that is not this machine is
accepted with a warning at start-up, since every state and image is then sent to another machine
(unencrypted over `http://`). An unreachable server, a timeout (60 s by default,
`--llama-server-timeout`), an HTTP error or a reply without the label probabilities is an error
naming the address and the cause: HTTP 502 from `serve`, a non-zero exit from `ask`.

## Limits

- A question may have at most as many options as `serve_config.json` has labels (255 in v1.3.1);
  more is an HTTP 422 (a non-zero exit for `ask`).
- A state longer than `serve_config.json`'s `max_state_tokens`, or a prompt longer than the
  model's context window, is refused rather than shortened.
- Each question is read as one whole prompt. In two-way reading the state's positions depend on
  the question that follows, so the state is not cached and shared between questions, and a
  request with many questions costs about that many forwards of the whole prompt.
- `--device mlx` (Apple silicon, the `mlx` extra) runs the same readout through mlx-lm. If MLX
  cannot load or run the model, the error says why and points to `--device cpu` (or `cuda`,
  `mps`). CI on Linux does not run it.
- On CPU the 4B weights are read in fp32, about 17 GB of memory.

`GET /health` returns the same keys as for a strands-decider checkpoint: `base_model` is the one
`serve_config.json` names, `num_slots` is the number of labels, `max_length` is the context window,
`temperature` is the per-type temperatures of the reading served, for example
`{"choice": 0.75, "noul": 0.8, "score": 0.65}`, and `device` is `llama-server` with
`--engine llama-server`.

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

## Timing it

`evaluation/bench_local.py` times requests on any of the builds: `--attention`, `--revision`,
`--image FILE` (repeatable; the CSV's `images` column gives the number), and
`--engine llama-server` with `--llama-server-url`. An h2o-lightning checkpoint is timed on one
path (`single`), since each question is one whole prompt.

```bash
python evaluation/bench_local.py h2oai/h2o-lightning-4b --revision REV --device cuda \
  --attention one-way --image page.png --lengths 256,1024 --questions 1,4
```

## Checking the numbers

The tests use tiny random models, so they cannot show that the real weights give the answers
h2o's own serving gives. [`evaluation/h2o_lightning/run_checks.sh`](../evaluation/h2o_lightning/README.md)
does, on a GPU Linux machine, a CPU Linux machine and a Mac, installed the way a user installs
strands-decider: it runs the Python examples above, `ask` and `serve`, compares the answers of
each build (`evaluation/h2o_parity.py`) and times them (`evaluation/bench_local.py`). Its README
gives the rows, who runs each and the pass criteria, which were fixed before any result was seen.

The 12 text requests it sends, one JSON body per line (4 yes/no, 4 choice, 4 score):

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
