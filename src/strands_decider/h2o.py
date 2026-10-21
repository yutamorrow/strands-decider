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
    SERVE_CONFIG,
    LABEL_ROWS,
)
ATTENTION_MODES = ("two-way", "one-way")
DEFAULT_ATTENTION = "two-way"
QUESTION_TYPES = ("choice", "noul", "score")
_TEXT_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")
# Lower bound of a label's log-probability, as h2o's readout floors it.
_LOGPROB_FLOOR = math.log(1e-12)
_IMAGES_UNSUPPORTED = "h2o-lightning checkpoints are served text-only by strands-decider; image input is not supported"


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


def resolve(checkpoint: str, revision: str | None = None) -> str | None:
    """The local directory of an h2o-lightning checkpoint, or None for any other checkpoint.

    A Hub repo is downloaded with `allow_patterns` limited to `H2O_FILES` at `revision`
    (None: the repo's default branch, as huggingface_hub resolves it). A failed download is a
    FileNotFoundError naming the repo and the cause.
    """
    if not is_h2o(checkpoint, revision):
        return None
    if os.path.isdir(checkpoint):
        return checkpoint
    try:
        return str(snapshot_download(checkpoint, revision=revision, allow_patterns=list(H2O_FILES)))
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
        )
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
    tokenizer: Any, config: H2OServeConfig, state: Content, question: Question
) -> RenderedH2O:
    """One question's whole prompt, as `serve_config.json` fixes it: the chat template with
    thinking off over [system, user], then the answer prefill. The user turn is
    `record: ...`, `question: ...`, `options:` and one `{label}) {name}: {description}` line per
    option (or the markdown layout)."""
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
    user_text = user.strip()
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
    start += len(head)
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


def forward_layers(torso: Any, input_ids: torch.Tensor, *, bidirectional: bool) -> torch.Tensor:
    """The text model's forward, layer by layer: embed, every decoder layer, final norm. Only the
    full-attention layers' mask differs between the two readings."""
    hidden = torso.embed_tokens(input_ids)
    position_embeddings, positions = _rotary(torso, hidden)
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
    the output embedding rows for some token ids."""

    device: str
    hidden_size: int

    def last_hidden(self, ids: list[int], *, bidirectional: bool) -> torch.Tensor: ...

    def output_rows(self, token_ids: list[int]) -> torch.Tensor: ...


class TorchBackend:
    """A Qwen3.5 causal LM on a torch device."""

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
    return a `SystemOneResponse`. Build one with `load_h2o_engine` (torch) or
    `h2o_mlx.load` (MLX).
    """

    def __init__(
        self,
        *,
        serve_config: H2OServeConfig,
        label_rows: LabelRows,
        tokenizer: Any,
        backend: Backend,
        attention: str,
        config: H2OEngineConfig,
        max_length: int,
    ) -> None:
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
        label_rows.check_labels(serve_config.labels)
        if label_rows.hidden != backend.hidden_size:
            raise ValueError(
                f"label rows: hidden size {label_rows.hidden}, the model's is {backend.hidden_size}"
            )
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

    def label_logits(self, ids: list[int], token_ids: list[int]) -> dict[int, float]:
        """{token id: logit} at the answer slot for the reading served."""
        two_way = self.attention == "two-way"
        hidden = self.backend.last_hidden(ids, bidirectional=two_way)
        rows = (
            self.label_rows.rows_for(token_ids) if two_way else self.backend.output_rows(token_ids)
        )
        logits = (rows.to(torch.float64) @ hidden.to(torch.float64)).tolist()
        # The label-row file's softcap belongs to the two-way readout (h2o's pooled_readout);
        # h2o's one-way path reads the causal LM's own logprobs, which carry no such cap.
        cap = self.label_rows.softcap if two_way else 0.0
        if cap:
            logits = [cap * math.tanh(x / cap) for x in logits]
        return dict(zip(token_ids, logits, strict=True))

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.images:
            raise ValueError(_IMAGES_UNSUPPORTED)
        rendered = {
            name: render_prompt(self.tok, self.serve_config, request.state, q)
            for name, q in request.questions.items()
        }
        prompts = {name: self.tokenize(r) for name, r in rendered.items()}
        temperatures = self.serve_config.temperatures(self.attention)
        floor = self.serve_config.floor(self.attention)
        answers: dict[str, Answer] = {}
        for name, r in rendered.items():
            groups = [(label, self.label_rows.group(label)) for label in r.labels]
            token_ids = [t for _, g in groups for t in g]
            logits = self.label_logits(prompts[name], token_ids)
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


def read_checkpoint(path: str) -> tuple[H2OServeConfig, LabelRows, Any, Any, int]:
    """(serve config, label rows, tokenizer, text config, prompt window) of a local checkpoint."""
    from transformers import AutoConfig, AutoTokenizer

    serve_config = H2OServeConfig.from_file(os.path.join(path, SERVE_CONFIG))
    rows_path = os.path.join(path, serve_config.label_rows)
    if not os.path.exists(rows_path):
        raise FileNotFoundError(
            f"{rows_path}: missing; an h2o-lightning checkpoint ships its label rows"
        )
    label_rows = LabelRows.read(rows_path)
    label_rows.check_labels(serve_config.labels)
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
    attention: str = DEFAULT_ATTENTION,
    attn_implementation: str | None = None,
    model_name: str | None = None,
) -> H2OEngine:
    """An `H2OEngine` on a torch device for the local h2o-lightning checkpoint `path`
    (`resolve` gives it for a Hub id)."""
    import transformers

    if attention not in ATTENTION_MODES:
        raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
    implementation = attn_implementation or "sdpa"
    if attention == "two-way" and implementation not in ("sdpa", "eager"):
        raise ValueError(
            f"two-way attention needs attn_implementation sdpa or eager, not {implementation!r}"
        )
    serve_config, label_rows, tokenizer, text_cfg, max_length = read_checkpoint(path)
    if str(device).startswith("mps"):
        from .mps_kernels import install

        install()
    elif str(device) == "cpu":
        from .cpu_kernels import install as install_cpu

        install_cpu()
    # Half precision is slower than fp32 on CPU (see SystemOneEngine._upcast_torso_for_cpu).
    dtype: Any = torch.float32 if str(device) == "cpu" else "auto"
    lm: Any = transformers.Qwen3_5ForCausalLM.from_pretrained(
        path, config=text_cfg, dtype=dtype, attn_implementation=implementation
    )
    lm.to(device).eval()
    return H2OEngine(
        serve_config=serve_config,
        label_rows=label_rows,
        tokenizer=tokenizer,
        backend=TorchBackend(lm, device),
        attention=attention,
        config=H2OEngineConfig(device=device, model_name=model_name or serve_config.model),
        max_length=max_length,
    )
