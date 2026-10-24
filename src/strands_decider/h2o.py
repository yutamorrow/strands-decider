"""Serving h2oai/h2o-lightning-4b checkpoints through `ask` and `serve`.

An h2o-lightning checkpoint is a plain Qwen3.5 causal LM (`config.json`, `model.safetensors`
and its tokenizer) plus two serving files: `serve_config.json`, which fixes the prompt, the
option labels and the per-type temperatures, and `label_rows.lightning`, which lists the token
ids each option label is read from (and their lm_head rows). It has no
`strands_decider_config.json` and no slot or pointer head: the answer is read from the language
model's own label-token logits at the slot after the prompt's `Answer:` prefill.

Two readings of the same weights are supported:

- two-way (the default): the full-attention layers attend in both directions, the Gated DeltaNet
  layers stay causal; the final hidden state (after the final norm) at the last position is
  multiplied with the label rows of `label_rows.lightning` (and capped by the file's softcap,
  when it sets one).
- one-way: the ordinary causal forward; the same hidden state is multiplied with the model's
  output embedding rows for the same token ids (the causal LM's own logits; no softcap).

In both, each label's logits are pooled with logsumexp over its token ids and normalised over
the asked labels, then divided by the type's temperature (`serve_config.json`; the `two_way`
block's own values in two-way, falling back to the one-way ones) and softmaxed. A yes/no answer
under the configured floor moves to the floor on its own side.

Every question is rendered as one whole prompt, tokenised once and forwarded once: in two-way the
state's positions depend on what follows them, so no state cache is shared between questions.

Images (`serve --vision`, `ask --image`, `load_engine(..., vision=True)`; torch devices only) are
read by h2o_vision.py: one placeholder per image opens the user turn, and an image request uses
the `image` block's temperature (`two_way.image_temperature` in two-way when set).

The light builds read one-way by default: an MLX-quantised build on `--device mlx` (h2o_mlx.py)
and the GGUF build through a llama-server (`--engine llama-server`, h2o_llama.py).

The existing strands-decider checkpoint path (`StrandsDeciderModel`, `SystemOneEngine`) is
untouched; `infer.load_engine` and `server.create_app` branch here only when `resolve` finds an
h2o-lightning checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from .modeling import CONFIG_NAME, LEGACY_CONFIG_NAME
from .schema import (
    Answer,
    ChoiceAnswer,
    Content,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    derive_confidence,
    derive_score_confidence,
)

SERVE_CONFIG = "serve_config.json"
LABEL_ROWS = "label_rows.lightning"
# The label-row file's format name (its safetensors metadata `format`).
LABEL_ROWS_FORMAT = "lightning-label-rows-1"
# Exactly the files an h2o-lightning checkpoint is read from. The repo also holds the h2o shim,
# its adapter and assets, which this package never runs and so never downloads. A name missing
# from the repo is simply not downloaded.
H2O_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    SERVE_CONFIG,
    LABEL_ROWS,
)
# The files the prompt and the readout need when the weights are elsewhere (a llama-server
# running the GGUF build): everything but the weights and the label rows.
H2O_CONFIG_FILES = tuple(f for f in H2O_FILES if f not in ("model.safetensors", LABEL_ROWS))
ATTENTION_MODES = ("two-way", "one-way")
DEFAULT_ATTENTION = "two-way"
# Where the weights run: in this process (torch or MLX), or in a llama-server (h2o_llama.py).
ENGINES = ("local", "llama-server")
# The default reading of the light builds (MLX-quantised, GGUF through llama-server), which
# read one-way only unless the checkpoint ships label rows.
LIGHT_DEFAULT_ATTENTION = "one-way"
QUESTION_TYPES = ("choice", "noul", "score")
_TEXT_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
# Lower bound of a label's log-probability, as h2o's readout floors it.
_LOGPROB_FLOOR = math.log(1e-12)
# Qwen's image placeholder, expanded to one `<|image_pad|>` per image token before tokenising.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
MLX_IMAGES_UNSUPPORTED = (
    "image input is not supported on --device mlx; use a torch device (cuda, mps or cpu) for images"
)
_IMAGES_NOT_LOADED = (
    "this h2o-lightning engine was loaded without image input; serve with --vision "
    "(ask --image, or load_engine(..., vision=True)) for images"
)
_QUANTISED_TORCH = (
    "{path}: an MLX-quantised h2o-lightning checkpoint (config.json sets quantization); "
    "it is read only with --device mlx"
)
# Formats whose support was stopped because a premise of it failed (FR3x): name -> the reason,
# which a load of that format reports. Names: "mlx-quantised", "llama-server",
# "llama-server-images". Empty: every format is supported.
STOPPED_FORMATS: dict[str, str] = {}


def check_not_stopped(name: str) -> None:
    """Refuse a format listed in `STOPPED_FORMATS`, giving the reason recorded there."""
    reason = STOPPED_FORMATS.get(name)
    if reason is not None:
        raise ValueError(
            f"{name} is not supported by strands-decider: {reason} (docs/h2o-lightning.md)"
        )


# ------------------------------------------------------------------------------ detection
def _local_is_h2o(path: str) -> bool:
    has = lambda name: os.path.exists(os.path.join(path, name))  # noqa: E731
    return has(SERVE_CONFIG) and not has(CONFIG_NAME) and not has(LEGACY_CONFIG_NAME)


def _hub_has(repo: str, filename: str, revision: str | None) -> bool:
    """True when `filename` can be fetched from the Hub repo (or the cache, offline)."""
    try:
        hf_hub_download(repo, filename, revision=revision)
    except EntryNotFoundError:  # also LocalEntryNotFoundError: offline and not cached
        return False
    return True


def _hub_error(repo: str, what: str, exc: BaseException) -> FileNotFoundError:
    """A Hub failure on a repo already found to be h2o-lightning, naming the repo and the cause
    (the same shape as `modeling.checkpoint_dir`'s)."""
    return FileNotFoundError(
        f"{repo}: an h2o-lightning checkpoint (it has {SERVE_CONFIG}), but {what} failed on the "
        f"Hugging Face Hub: {type(exc).__name__}: {exc}"
    )


def is_h2o(checkpoint: str, revision: str | None = None) -> bool:
    """True when `checkpoint` (a local directory or a Hub repo id) is an h2o-lightning checkpoint:
    it has `serve_config.json` and neither `strands_decider_config.json` nor `hobson_config.json`.

    For a Hub id only `serve_config.json` (and, when it exists, the absence of the two decider
    configs) is checked, never the whole repo. Any failure to fetch `serve_config.json` (no such
    file, no such repo, offline, not a repo id) answers False, so the existing loader runs exactly
    as before and reports its own error. Once `serve_config.json` is found, a failure other than
    "no such file" while checking for the decider configs is a FileNotFoundError naming the repo
    and the cause.
    """
    if os.path.isdir(checkpoint):
        return _local_is_h2o(checkpoint)
    try:
        if not _hub_has(checkpoint, SERVE_CONFIG, revision):
            return False
    except Exception:  # any failure here leaves the existing path to report it
        return False
    try:
        return not any(
            _hub_has(checkpoint, name, revision) for name in (CONFIG_NAME, LEGACY_CONFIG_NAME)
        )
    except Exception as e:
        raise _hub_error(checkpoint, f"checking for {CONFIG_NAME} / {LEGACY_CONFIG_NAME}", e) from e


def _config_is_quantised(config_file: str) -> bool:
    with open(config_file, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return isinstance(cfg, dict) and any(
        cfg.get(k) for k in ("quantization", "quantization_config")
    )


def is_mlx_quantised(path: str, revision: str | None = None) -> bool:
    """True when the h2o-lightning checkpoint `path` (a local directory or a Hub repo id) is an
    MLX-quantised build: its `config.json` sets `quantization` (or `quantization_config`). For a
    Hub id only `config.json` is fetched; a missing one answers False."""
    if os.path.isdir(path):
        config_file = os.path.join(path, "config.json")
        return os.path.exists(config_file) and _config_is_quantised(config_file)
    try:
        config_file = hf_hub_download(path, "config.json", revision=revision)
    except EntryNotFoundError:
        return False
    except Exception as e:
        raise _hub_error(path, "fetching config.json", e) from e
    return _config_is_quantised(config_file)


def resolve(
    checkpoint: str,
    revision: str | None = None,
    *,
    files: Sequence[str] = H2O_FILES,
    refuse_quantised: bool = False,
) -> str | None:
    """The local directory of an h2o-lightning checkpoint, or None for any other checkpoint.

    A Hub repo is downloaded with `allow_patterns` limited to `files` (`H2O_FILES`: every file the
    checkpoint is read from) at `revision` (None: the repo's default branch, as huggingface_hub
    resolves it). A failed download is a FileNotFoundError naming the repo and the cause. With
    `refuse_quantised` (a torch device), an MLX-quantised checkpoint is refused from its
    `config.json` before the weights are downloaded.
    """
    if not is_h2o(checkpoint, revision):
        return None
    if refuse_quantised and is_mlx_quantised(checkpoint, revision):
        raise ValueError(_QUANTISED_TORCH.format(path=checkpoint))
    if os.path.isdir(checkpoint):
        return checkpoint
    try:
        return str(snapshot_download(checkpoint, revision=revision, allow_patterns=list(files)))
    except Exception as e:
        raise _hub_error(checkpoint, "downloading its files", e) from e


# ------------------------------------------------------------------------- serve_config.json
def _temperatures(value: Any, where: str) -> dict[str, float]:
    if isinstance(value, dict):
        if set(value) != set(QUESTION_TYPES):
            raise ValueError(
                f"{where} by type must give exactly {list(QUESTION_TYPES)}, got {sorted(value)}"
            )
        return {k: float(value[k]) for k in QUESTION_TYPES}
    return {k: float(value) for k in QUESTION_TYPES}


@dataclass
class H2OServeConfig:
    """The serving settings of `serve_config.json` this package uses. Unknown keys are ignored."""

    model: str
    base_model: str
    labels: list[str]
    temperature: dict[str, float]
    noul_floor: float
    two_way_temperature: dict[str, float]
    two_way_noul_floor: float
    chat_template: bool = True
    system: str | None = None
    layout: str = "plain"
    state_compact: bool = True
    prefill: str = "Answer:"
    max_state_tokens: int = 32000
    default_instructions: str = "Answer the question below."
    label_rows: str = LABEL_ROWS
    # The `image` block: None when serve_config.json has none (the checkpoint takes no images).
    image_temperature: float | None = None
    two_way_image_temperature: float | None = None
    max_images: int = 0
    max_pixels: int = 0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, source: str = SERVE_CONFIG) -> H2OServeConfig:
        missing = [k for k in ("prompt", "labels", "temperature", "noul_floor") if k not in cfg]
        if missing:
            raise ValueError(f"{source}: missing {missing}")
        prompt = cfg["prompt"]
        if not isinstance(prompt, dict):
            raise ValueError(f"{source}: `prompt` must be an object")
        if prompt.get("enable_thinking", False):
            raise ValueError(f"{source}: prompt.enable_thinking must be false")
        layout = str(prompt.get("layout", "plain"))
        if layout not in ("plain", "markdown"):
            raise ValueError(f"{source}: prompt.layout {layout!r} is not supported")
        if not isinstance(cfg["labels"], list) or not cfg["labels"]:
            raise ValueError(f"{source}: `labels` must be the non-empty list of option labels")
        labels = [str(x) for x in cfg["labels"]][: int(cfg.get("max_options", len(cfg["labels"])))]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{source}: the labels are not distinct")
        temperature = _temperatures(cfg["temperature"], f"{source} temperature")
        noul_floor = float(cfg["noul_floor"])
        two_way_raw = cfg.get("two_way")
        two_way: dict[str, Any] = two_way_raw if isinstance(two_way_raw, dict) else {}
        tw_t = two_way.get("temperature")
        tw_f = two_way.get("noul_floor")
        image_raw = cfg.get("image")
        if image_raw is not None and not isinstance(image_raw, dict):
            raise ValueError(f"{source}: `image` must be an object")
        image: dict[str, Any] = image_raw or {}
        image_t, tw_image_t = image.get("temperature"), two_way.get("image_temperature")
        out = cls(
            model=str(cfg.get("model") or "h2o-lightning"),
            base_model=str(cfg.get("base_model") or cfg.get("model") or ""),
            labels=labels,
            temperature=temperature,
            noul_floor=noul_floor,
            two_way_temperature=(
                temperature
                if tw_t is None
                else _temperatures(tw_t, f"{source} two_way.temperature")
            ),
            two_way_noul_floor=noul_floor if tw_f is None else float(tw_f),
            chat_template=bool(prompt.get("chat_template", True)),
            system=prompt.get("system") or None,
            layout=layout,
            state_compact=bool(prompt.get("state_compact", True)),
            prefill=str(prompt.get("prefill", "Answer:")),
            max_state_tokens=int(prompt.get("max_state_tokens", 32000)),
            default_instructions=str(
                prompt.get("default_instructions", "Answer the question below.")
            ),
            label_rows=str(two_way.get("label_rows", LABEL_ROWS)),
            image_temperature=None if image_t is None else float(image_t),
            two_way_image_temperature=None if tw_image_t is None else float(tw_image_t),
            max_images=int(image.get("max_images", 1)) if image_raw is not None else 0,
            max_pixels=int(image.get("max_pixels", 4096 * 4096)) if image_raw is not None else 0,
        )
        if image_raw is not None and (out.max_images < 1 or out.max_pixels < 1):
            raise ValueError(f"{source}: image.max_images and image.max_pixels must be positive")
        for name, value in (
            ("image.temperature", out.image_temperature),
            ("two_way.image_temperature", out.two_way_image_temperature),
        ):
            if value is not None and not 0.05 <= value <= 20.0:
                raise ValueError(f"{source}: {name} {value} is outside [0.05, 20]")
        for name, t in (
            ("temperature", out.temperature),
            ("two_way.temperature", out.two_way_temperature),
        ):
            if not all(0.05 <= x <= 20.0 for x in t.values()):
                raise ValueError(f"{source}: {name} {t} is outside [0.05, 20]")
        for name, f in (
            ("noul_floor", out.noul_floor),
            ("two_way.noul_floor", out.two_way_noul_floor),
        ):
            if f and not 0.5 < f < 1.0:
                raise ValueError(f"{source}: {name} {f} must be in (0.5, 1), or 0 to turn it off")
        return out

    @classmethod
    def from_file(cls, path: str) -> H2OServeConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), source=path)

    def temperatures(self, attention: str) -> dict[str, float]:
        return self.two_way_temperature if attention == "two-way" else self.temperature

    def floor(self, attention: str) -> float:
        return self.two_way_noul_floor if attention == "two-way" else self.noul_floor

    @property
    def takes_images(self) -> bool:
        """Whether serve_config.json has an `image` block (no block: the checkpoint takes none)."""
        return self.max_images > 0

    def request_temperatures(self, attention: str, with_images: bool) -> dict[str, float]:
        """The per-type temperatures of one request. With images, every type uses the image
        temperature: `two_way.image_temperature` in two-way when set, else `image.temperature`;
        without either, the per-type ones."""
        if with_images:
            image_t = (
                self.two_way_image_temperature
                if attention == "two-way" and self.two_way_image_temperature is not None
                else self.image_temperature
            )
            if image_t is not None:
                return {k: image_t for k in QUESTION_TYPES}
        return self.temperatures(attention)


# ------------------------------------------------------------------------- label_rows.lightning
@dataclass
class LabelRows:
    """`label_rows.lightning`: `W` fp32 [n, hidden] (one lm_head row per token id), `token_ids`
    int64 [n], and in the metadata the label groups `[[label, [token ids]], ...]`."""

    token_ids: list[int]
    rows: torch.Tensor
    groups: list[tuple[str, list[int]]]
    hidden: int
    softcap: float = 0.0
    index: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {t: k for k, t in enumerate(self.token_ids)}
        if len(self.index) != len(self.token_ids):
            raise ValueError("label rows: a token id appears twice")
        labels = [label for label, _ in self.groups]
        if len(set(labels)) != len(labels):
            raise ValueError("label rows: the labels are not distinct")
        empty = [label for label, ids in self.groups if not ids]
        if empty:
            raise ValueError(f"label rows: label(s) {empty} have no token id")
        seen: dict[int, str] = {}
        for label, ids in self.groups:
            for t in ids:
                if t in seen:
                    raise ValueError(
                        f"label rows: token id {t} belongs to labels {seen[t]!r} and {label!r}"
                    )
                seen[t] = label
                if t not in self.index:
                    raise ValueError(f"label rows: no row for token id {t} of label {label!r}")

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.groups]

    def group(self, label: str) -> list[int]:
        for name, ids in self.groups:
            if name == label:
                return ids
        raise ValueError(f"label rows: no label {label!r}")

    def rows_for(self, token_ids: Sequence[int]) -> torch.Tensor:
        return self.rows[[self.index[t] for t in token_ids]]

    def check_labels(self, labels: Sequence[str]) -> None:
        """The file's labels must start with `labels`, in that order (serve_config.json's)."""
        have = self.labels[: len(labels)]
        if have != list(labels):
            raise ValueError(
                f"label rows: the file's labels {have[:6]}... are not serve_config.json's "
                f"{list(labels)[:6]}... in that order"
            )

    @classmethod
    def read(cls, path: str) -> LabelRows:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            if meta.get("format") != LABEL_ROWS_FORMAT:
                raise ValueError(
                    f"{path}: not a {LABEL_ROWS_FORMAT} file (format {meta.get('format')!r})"
                )
            names = set(f.keys())
            if not {"W", "token_ids"} <= names:
                raise ValueError(f"{path}: needs the tensors 'W' and 'token_ids'")
            w = f.get_tensor("W")
            ids = f.get_tensor("token_ids")
        if w.dtype != torch.float32 or w.dim() != 2:
            raise ValueError(f"{path}: 'W' must be a 2-d float32 tensor")
        if ids.dtype != torch.int64 or ids.dim() != 1 or ids.numel() != w.shape[0]:
            raise ValueError(
                f"{path}: 'token_ids' must be a 1-d int64 tensor with one id per row of 'W'"
            )
        hidden = int(meta.get("hidden", w.shape[1]))
        if hidden != w.shape[1]:
            raise ValueError(
                f"{path}: the hidden metadata {hidden} does not match 'W' {tuple(w.shape)}"
            )
        sc = meta.get("softcap", "none")
        softcap = 0.0 if sc in ("none", "", None) else float(sc)
        token_ids = [int(x) for x in ids.tolist()]
        if "groups" in meta:
            raw = json.loads(meta["groups"])
            groups = [(str(label), [int(t) for t in g]) for label, g in raw]
        else:  # one token per label, in label order
            order = json.loads(meta.get("label_order", "[]"))
            if len(order) != len(token_ids):
                raise ValueError(
                    f"{path}: no groups metadata and the label order does not match the ids"
                )
            groups = [(str(label), [t]) for label, t in zip(order, token_ids, strict=True)]
        return cls(token_ids=token_ids, rows=w, groups=groups, hidden=hidden, softcap=softcap)


# ------------------------------------------------------------------------------ prompt
def _desc_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_state_text(state: Content, compact: bool) -> str:
    """The record: a string verbatim, anything else as JSON (compact when configured)."""
    if isinstance(state, str):
        return state
    if compact:
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(state, indent=1)


def _instructions_text(value: Content) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=1)


def question_options(question: Question) -> tuple[list[str], list[str]]:
    """(option names, descriptions). A yes/no question is always [true, false]; a score's
    options are its levels' indices, in order."""
    if question.type == "noul":
        criteria = question.criteria or {}
        true_desc, false_desc = criteria.get("true"), criteria.get("false")
        return ["true", "false"], [
            _desc_text(true_desc) if true_desc not in (None, "") else "the statement holds",
            _desc_text(false_desc) if false_desc not in (None, "") else "it does not",
        ]
    if question.type == "score":
        return [str(i) for i in range(len(question.criteria))], [
            _desc_text(v) for v in question.criteria
        ]
    names = list(question.criteria)
    return names, [
        _desc_text(question.criteria[n]) if question.criteria[n] is not None else n for n in names
    ]


@dataclass
class RenderedH2O:
    text: str
    kind: str
    names: list[str]
    descs: list[str]
    labels: list[str]
    # Character range of the state (the record) inside `text`.
    state_span: tuple[int, int]


def render_prompt(
    tokenizer: Any,
    config: H2OServeConfig,
    state: Content,
    question: Question,
    *,
    n_images: int = 0,
    image_placeholder: str = IMAGE_PLACEHOLDER,
) -> RenderedH2O:
    """One question's whole prompt, as `serve_config.json` fixes it: the chat template with
    thinking off over [system, user], then the answer prefill. The user turn is
    `record: ...`, `question: ...`, `options:` and one `{label}) {name}: {description}` line per
    option (or the markdown layout). With `n_images`, one `image_placeholder` per image opens the
    user turn, before the record (as h2o's own serving places images)."""
    names, descs = question_options(question)
    if len(names) > len(config.labels):
        raise ValueError(
            f"question has {len(names)} options but this model has {len(config.labels)} labels"
        )
    labels = config.labels[: len(names)]
    state_text = render_state_text(state, config.state_compact)
    instructions = _instructions_text(question.instructions).lstrip() or config.default_instructions
    body = "\n".join(f"{lab}) {n}: {d}" for lab, n, d in zip(labels, names, descs, strict=True))
    if config.layout == "plain":
        head = "record: "
        user = f"{head}{state_text}\nquestion: {instructions}\noptions:\n{body}"
    else:
        head = "### RECORD\n"
        user = f"{head}{state_text}\n\n### QUESTION\n{instructions}\n\n### OPTIONS\n{body}\n"
    user_text = image_placeholder * n_images + user.strip()
    system = config.system.strip() if config.system is not None else None
    if config.chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "serve_config.json uses the chat template, but the tokenizer has none "
                "(chat_template.jinja missing from the checkpoint?)"
            )
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": user_text})
        text = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            + config.prefill
        )
        after = 0 if system is None else text.find(system) + len(system)
    else:
        text = (
            "\n\n".join(([] if system is None else [system]) + [user_text]) + f"\n{config.prefill}"
        )
        after = 0 if system is None else len(system)
    start = text.find(user_text, after)
    if start < 0:
        raise ValueError("the chat template does not contain the user turn verbatim")
    start += len(image_placeholder) * n_images + len(head)
    return RenderedH2O(
        text=text,
        kind=question.type,
        names=names,
        descs=descs,
        labels=labels,
        state_span=(start, start + len(state_text)),
    )


# ------------------------------------------------------------------------------ readout
def group_log_probs(
    logits: dict[int, float], groups: Sequence[tuple[str, Sequence[int]]]
) -> list[float]:
    """log P(label) for each label of `groups`: logsumexp over the label's token ids,
    renormalised over these labels, floored at log(1e-12)."""
    acc = []
    for label, ids in groups:
        v = [logits[i] for i in ids]
        if not v or not all(math.isfinite(x) for x in v):
            raise ValueError(f"an invalid logit for label {label!r}")
        acc.append(_logsumexp(v))
    z = _logsumexp(acc)
    return [max(a - z, _LOGPROB_FLOOR) for a in acc]


def _logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(x - m) for x in values))


def probabilities(log_probs: Sequence[float], temperature: float) -> list[float]:
    """softmax(log-probabilities / T) over the options."""
    v = [x / max(float(temperature), 1e-3) for x in log_probs]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def commit_noul(p_true: float, floor: float) -> float:
    """A yes/no probability with max(P, 1 - P) under `floor` moves to the floor on its own side
    (a tie goes to true). The answer never changes; a floor of 0 turns this off."""
    if not floor or max(p_true, 1.0 - p_true) >= floor:
        return p_true
    return floor if p_true >= 0.5 else 1.0 - floor


def to_answer(rendered: RenderedH2O, probs: Sequence[float], noul_floor: float) -> Answer:
    """The answer in the same shape a strands-decider checkpoint returns."""
    p = list(probs)
    if rendered.kind == "noul":
        return NoulAnswer(noul=round(commit_noul(p[0], noul_floor), 4))
    if rendered.kind == "choice":
        best = max(range(len(p)), key=p.__getitem__)  # the first maximum
        return ChoiceAnswer(
            choice=rendered.names[best],
            probabilities={n: round(x, 4) for n, x in zip(rendered.names, p, strict=True)},
            confidence=round(derive_confidence(p), 4),
        )
    return ScoreAnswer(
        score=round(sum(i * x for i, x in enumerate(p)), 4),
        legend={str(i): d for i, d in enumerate(rendered.descs)},
        probabilities={str(i): round(x, 4) for i, x in enumerate(p)},
        confidence=round(derive_score_confidence(p), 4),
    )


# ------------------------------------------------------------------------------ torch forward
def _full_attention_mask(
    length: int, bidirectional: bool, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """An additive [1, 1, L, L] mask for the full-attention layers: all zeros (two-way) or the
    causal mask. One unpadded row, so there is no padding to hide."""
    mask = torch.zeros(1, 1, length, length, dtype=dtype, device=device)
    if not bidirectional:
        upper = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
        mask = mask.masked_fill(upper, torch.finfo(dtype).min)
    return mask


def _rotary(torso: Any, hidden: torch.Tensor) -> tuple[Any, torch.Tensor]:
    positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, -1)
    return torso.rotary_emb(hidden, positions[None]), positions


def run_layer(torso: Any, index: int, hidden: torch.Tensor, *, bidirectional: bool) -> torch.Tensor:
    """Decoder layer `index` of a Qwen3.5 text model on `hidden` [1, L, d], from position 0. A
    full-attention layer gets the two-way or the causal mask; a Gated DeltaNet layer is causal
    by construction and gets no mask."""
    layer = torso.layers[index]
    kind = torso.config.layer_types[index]
    position_embeddings, positions = _rotary(torso, hidden)
    mask = (
        _full_attention_mask(hidden.shape[1], bidirectional, hidden.dtype, hidden.device)
        if kind == "full_attention"
        else None
    )
    out: torch.Tensor = layer(
        hidden, position_embeddings=position_embeddings, attention_mask=mask, position_ids=positions
    )
    return out


def forward_layers(
    torso: Any,
    input_ids: torch.Tensor | None = None,
    *,
    bidirectional: bool,
    inputs_embeds: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """The text model's forward, layer by layer: embed, every decoder layer, final norm. Only the
    full-attention layers' mask differs between the two readings.

    `inputs_embeds` replaces the embedding of `input_ids` (an image prompt, whose image tokens
    carry the vision tower's features). `position_ids` [4, 1, L] are the text row and the three
    M-RoPE rows of an image prompt (`get_rope_index`); without them, positions are 0..L-1."""
    hidden = torso.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    if position_ids is None:
        position_embeddings, positions = _rotary(torso, hidden)
    else:
        positions = position_ids[0]
        position_embeddings = torso.rotary_emb(hidden, position_ids[1:])
    mask = _full_attention_mask(hidden.shape[1], bidirectional, hidden.dtype, hidden.device)
    for i, layer in enumerate(torso.layers):
        full = torso.config.layer_types[i] == "full_attention"
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=mask if full else None,
            position_ids=positions,
        )
    out: torch.Tensor = torso.norm(hidden)
    return out


class Backend(Protocol):
    """What `H2OEngine` needs from a torso: the final hidden state at the last position, and
    the output embedding rows for some token ids. `supports_images`: whether `prepare_images`
    and `last_hidden(..., images=)` work; `supports_two_way`: whether two-way can be read."""

    device: str
    hidden_size: int
    supports_images: bool
    supports_two_way: bool

    def last_hidden(self, ids: list[int], *, bidirectional: bool) -> torch.Tensor: ...

    def output_rows(self, token_ids: list[int]) -> torch.Tensor: ...


class TorchBackend:
    """A Qwen3.5 causal LM on a torch device."""

    supports_images = False
    supports_two_way = True

    def __init__(self, lm: Any, device: str) -> None:
        self.lm = lm
        self.torso = lm.model
        self.device = device
        self.hidden_size = int(self.torso.config.hidden_size)

    def last_hidden(self, ids: list[int], *, bidirectional: bool) -> torch.Tensor:
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=self.device)
            if bidirectional:
                hidden = forward_layers(self.torso, input_ids, bidirectional=True)
            else:
                hidden = self.torso(
                    input_ids=input_ids, use_cache=False, return_dict=True
                ).last_hidden_state
            out: torch.Tensor = hidden[0, -1].to(device="cpu", dtype=torch.float32)
        return out

    def output_rows(self, token_ids: list[int]) -> torch.Tensor:
        with torch.inference_mode():
            weight = self.lm.get_output_embeddings().weight
            out: torch.Tensor = weight[token_ids].to(device="cpu", dtype=torch.float32)
        return out


# ------------------------------------------------------------------------------ the engine
@dataclass
class H2OEngineConfig:
    device: str = "cuda"
    model_name: str = "h2o-lightning"
    # Not used by this engine (no state is cached); kept so /health reports the same keys.
    use_prefix_cache: bool = False


class H2OEngine:
    """Answers System One requests from an h2o-lightning checkpoint.

    The same surface as `SystemOneEngine`: `evaluate(request)` and `ask(state, questions)`
    return a `SystemOneResponse`. Build one with `load_h2o_engine` (torch), `h2o_mlx.load`
    (MLX) or `h2o_llama.load` (a llama-server running the GGUF build).

    A checkpoint without `label_rows.lightning` (the light builds) is read one-way only; its
    labels' token ids then come from the tokenizer (`label_groups_from_tokenizer`).
    """

    def __init__(
        self,
        *,
        serve_config: H2OServeConfig,
        label_rows: LabelRows | None,
        tokenizer: Any,
        backend: Backend,
        attention: str,
        config: H2OEngineConfig,
        max_length: int,
        label_groups: Sequence[tuple[str, list[int]]] | None = None,
    ) -> None:
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
        if attention == "two-way":
            if not backend.supports_two_way:
                raise ValueError(
                    f"{type(backend).__name__} reads one-way only; two-way attention is not "
                    "supported here (use --attention one-way)"
                )
            if label_rows is None:
                raise ValueError(
                    f"this checkpoint has no {LABEL_ROWS}, so it cannot be read two-way "
                    "(use --attention one-way)"
                )
        if label_rows is not None:
            label_rows.check_labels(serve_config.labels)
            if label_rows.hidden != backend.hidden_size:
                raise ValueError(
                    f"label rows: hidden size {label_rows.hidden}, the model's is "
                    f"{backend.hidden_size}"
                )
            groups = label_rows.groups
        else:
            groups = list(
                label_groups
                if label_groups is not None
                else label_groups_from_tokenizer(tokenizer, serve_config)
            )
        self._groups = {label: list(ids) for label, ids in groups}
        self.serve_config = serve_config
        self.label_rows = label_rows
        self.tok = tokenizer
        self.backend = backend
        self.attention = attention
        self.cfg = config
        self.max_length = max_length
        # A fast tokenizer is not safe to call from two threads at once (see SystemOneEngine).
        self._tok_lock = threading.Lock()

    def health_fields(self) -> dict[str, Any]:
        """/health's model fields: the base model, the number of labels, the prompt window and
        the per-type temperatures of the reading served."""
        return {
            "base_model": self.serve_config.base_model,
            "num_slots": len(self.serve_config.labels),
            "max_length": self.max_length,
            "temperature": dict(self.serve_config.temperatures(self.attention)),
        }

    def group(self, label: str) -> list[int]:
        """The token ids `label` is read from."""
        try:
            return self._groups[label]
        except KeyError:
            raise ValueError(f"no token ids for label {label!r}") from None

    def tokenize(self, rendered: RenderedH2O) -> list[int]:
        """The whole prompt tokenised once; refuses (never truncates) a state over
        `max_state_tokens` or a prompt over the model's window."""
        with self._tok_lock:
            enc = self.tok(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
        ids: list[int] = list(enc["input_ids"])
        a, b = rendered.state_span
        n_state = sum(1 for lo, hi in enc["offset_mapping"] if hi > lo and lo < b and hi > a)
        if n_state > self.serve_config.max_state_tokens:
            raise ValueError(
                f"state of {n_state} tokens exceeds serve_config.json max_state_tokens "
                f"{self.serve_config.max_state_tokens}"
            )
        if len(ids) > self.max_length:
            raise ValueError(
                f"prompt of {len(ids)} tokens exceeds the context window of {self.max_length} tokens"
            )
        return ids

    def label_logits(
        self, ids: list[int], token_ids: list[int], *, images: Any = None
    ) -> dict[int, float]:
        """{token id: logit} at the answer slot for the reading served. `images` is the
        backend's `prepare_images` result for an image request."""
        two_way = self.attention == "two-way"
        extra = {} if images is None else {"images": images}
        hidden = self.backend.last_hidden(ids, bidirectional=two_way, **extra)
        if two_way:
            assert self.label_rows is not None  # checked in __init__
            rows = self.label_rows.rows_for(token_ids)
        else:
            rows = self.backend.output_rows(token_ids)
        logits = (rows.to(torch.float64) @ hidden.to(torch.float64)).tolist()
        # The label-row file's softcap belongs to the two-way readout (h2o's pooled_readout);
        # h2o's one-way path reads the causal LM's own logprobs, which carry no such cap.
        cap = self.label_rows.softcap if two_way and self.label_rows is not None else 0.0
        if cap:
            logits = [cap * math.tanh(x / cap) for x in logits]
        return dict(zip(token_ids, logits, strict=True))

    def _prepare_images(self, images: Sequence[str]) -> Any:
        """Check, decode and shrink a request's images, then hand them to the backend."""
        if not self.backend.supports_images:
            raise ValueError(
                MLX_IMAGES_UNSUPPORTED if self.backend.device == "mlx" else _IMAGES_NOT_LOADED
            )
        sc = self.serve_config
        if not sc.takes_images:
            raise ValueError(
                f"this checkpoint's {SERVE_CONFIG} has no `image` block; it takes no images"
            )
        if len(images) > sc.max_images:
            raise ValueError(f"{len(images)} images; this checkpoint takes at most {sc.max_images}")
        from .h2o_vision import decode_images

        prepare: Any = getattr(self.backend, "prepare_images")  # noqa: B009 - Protocol-optional
        return prepare(decode_images(images, sc.max_pixels))

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        images = self._prepare_images(request.images) if request.images else None
        n_images = len(request.images) if request.images else 0
        # Where each image goes in the prompt: Qwen's placeholder, or the backend's own marker.
        placeholder = getattr(images, "placeholder", None) or IMAGE_PLACEHOLDER
        rendered = {
            name: render_prompt(
                self.tok,
                self.serve_config,
                request.state,
                q,
                n_images=n_images,
                image_placeholder=placeholder,
            )
            for name, q in request.questions.items()
        }
        tokens_per_image = getattr(images, "tokens_per_image", None)
        if tokens_per_image is not None:
            from .h2o_vision import expand_rendered

            rendered = {n: expand_rendered(r, tokens_per_image) for n, r in rendered.items()}
        prompts = {name: self.tokenize(r) for name, r in rendered.items()}
        temperatures = self.serve_config.request_temperatures(self.attention, images is not None)
        floor = self.serve_config.floor(self.attention)
        # A backend that reads the label logits itself (a llama-server) is given the prompt.
        reads_logits: Any = getattr(self.backend, "label_logits", None)
        answers: dict[str, Answer] = {}
        for name, r in rendered.items():
            groups = [(label, self.group(label)) for label in r.labels]
            token_ids = [t for _, g in groups for t in g]
            if reads_logits is not None:
                logits = reads_logits(r.text, prompts[name], token_ids, images=images)
            else:
                logits = self.label_logits(prompts[name], token_ids, images=images)
            probs = probabilities(group_log_probs(logits, groups), temperatures[r.kind])
            answers[name] = to_answer(r, probs, floor)
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(
                input_tokens=sum(len(p) for p in prompts.values()), output_tokens=len(answers)
            ),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))

    def ask_images(
        self, state: Content, questions: dict[str, Question], images: Sequence[str]
    ) -> SystemOneResponse:
        return self.evaluate(
            SystemOneRequest(state=state, questions=questions, images=list(images))
        )


def label_groups_from_tokenizer(
    tokenizer: Any, serve_config: H2OServeConfig
) -> list[tuple[str, list[int]]]:
    """Each label's token id when the checkpoint ships no `label_rows.lightning`: the single
    token that follows the `Answer:` prefill, as h2o builds its label rows ("labels via
    tokenizer.json after 'Answer:'"): the label with a leading space (` A`), else the bare label
    when that is the single token. A label that is neither is refused."""
    prefill = serve_config.prefill
    base = list(tokenizer(prefill, add_special_tokens=False)["input_ids"])
    groups: list[tuple[str, list[int]]] = []
    owner: dict[int, str] = {}
    for label in serve_config.labels:
        token: int | None = None
        for text in (" " + label, label):
            enc = list(tokenizer(prefill + text, add_special_tokens=False)["input_ids"])
            tail = enc[len(base) :] if enc[: len(base)] == base else []
            if len(tail) == 1:
                token = int(tail[0])
                break
        if token is None:
            raise ValueError(
                f"label {label!r} is not a single token after {prefill!r}; this checkpoint "
                f"needs its {LABEL_ROWS}"
            )
        if token in owner:
            raise ValueError(f"labels {owner[token]!r} and {label!r} share token id {token}")
        owner[token] = label
        groups.append((label, [token]))
    return groups


def read_checkpoint(
    path: str, *, require_label_rows: bool = True
) -> tuple[H2OServeConfig, LabelRows | None, Any, Any, int]:
    """(serve config, label rows, tokenizer, text config, prompt window) of a local checkpoint.
    Without `require_label_rows`, a checkpoint with no label-row file gives None for them."""
    from transformers import AutoConfig, AutoTokenizer

    serve_config = H2OServeConfig.from_file(os.path.join(path, SERVE_CONFIG))
    rows_path = os.path.join(path, serve_config.label_rows)
    label_rows: LabelRows | None = None
    if os.path.exists(rows_path):
        label_rows = LabelRows.read(rows_path)
        label_rows.check_labels(serve_config.labels)
    elif require_label_rows:
        raise FileNotFoundError(
            f"{rows_path}: missing; an h2o-lightning checkpoint ships its label rows"
        )
    base_cfg = AutoConfig.from_pretrained(path)
    if base_cfg.model_type not in _TEXT_MODEL_TYPES:
        raise ValueError(
            f"{path}: model_type {base_cfg.model_type!r}; h2o-lightning checkpoints are Qwen3.5"
        )
    text_cfg = base_cfg.get_text_config()
    max_length = int(
        getattr(text_cfg, "max_position_embeddings", None) or base_cfg.max_position_embeddings
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    return serve_config, label_rows, tokenizer, text_cfg, max_length


def load_h2o_engine(
    path: str,
    *,
    device: str = "cuda",
    attention: str | None = None,
    attn_implementation: str | None = None,
    model_name: str | None = None,
    vision: bool = False,
) -> H2OEngine:
    """An `H2OEngine` on a torch device for the local h2o-lightning checkpoint `path`
    (`resolve` gives it for a Hub id). `attention` None reads two-way. With `vision`, the
    vision tower is loaded too, so requests may carry `images` (h2o_vision.py). One-way also reads
    a checkpoint without `label_rows.lightning` (h2o's releases before v1.3 ship none), with the
    labels' token ids from the tokenizer; two-way needs the file."""
    import transformers

    attention = attention or DEFAULT_ATTENTION
    if attention not in ATTENTION_MODES:
        raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
    if is_mlx_quantised(path):
        raise ValueError(_QUANTISED_TORCH.format(path=path))
    implementation = attn_implementation or "sdpa"
    if attention == "two-way" and implementation not in ("sdpa", "eager"):
        raise ValueError(
            f"two-way attention needs attn_implementation sdpa or eager, not {implementation!r}"
        )
    serve_config, label_rows, tokenizer, text_cfg, max_length = read_checkpoint(
        path, require_label_rows=attention == "two-way"
    )
    if str(device).startswith("mps"):
        from .mps_kernels import install

        install()
    elif str(device) == "cpu":
        from .cpu_kernels import install as install_cpu

        install_cpu()
    # Half precision is slower than fp32 on CPU (see SystemOneEngine._upcast_torso_for_cpu).
    dtype: Any = torch.float32 if str(device) == "cpu" else "auto"
    backend: Backend
    if vision:
        from .h2o_vision import load_vision_backend

        backend = load_vision_backend(
            path, serve_config, device=device, dtype=dtype, attn_implementation=implementation
        )
    else:
        lm: Any = transformers.Qwen3_5ForCausalLM.from_pretrained(
            path, config=text_cfg, dtype=dtype, attn_implementation=implementation
        )
        lm.to(device).eval()
        backend = TorchBackend(lm, device)
    return H2OEngine(
        serve_config=serve_config,
        label_rows=label_rows,
        tokenizer=tokenizer,
        backend=backend,
        attention=attention,
        config=H2OEngineConfig(device=device, model_name=model_name or serve_config.model),
        max_length=max_length,
    )
