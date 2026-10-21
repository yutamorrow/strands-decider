"""h2o-lightning checkpoints through `ask` and `serve` (src/strands_decider/h2o.py).

A tiny random-weight Qwen3.5 text model (Gated DeltaNet and full-attention layers) is saved with
the two serving files an h2o-lightning checkpoint carries, `serve_config.json` and
`label_rows.lightning`, and a word-level tokenizer with a minimal Qwen-style chat template.
Nothing is downloaded: the Hub functions are replaced where a Hub id is used.
"""

from __future__ import annotations

import fnmatch
import json
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from huggingface_hub.errors import EntryNotFoundError  # noqa: E402

from strands_decider import h2o, infer  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

LABELS = ["A", "B", "C", "D"]
SYSTEM = "You answer typed questions about a record."
# The template Qwen3.5's renders for [system, user] with the generation prompt and thinking off.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] | trim }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
SPECIALS = ["<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
SERVE_CONFIG = {
    "model": "h2o-lightning-test",
    "base_model": "Qwen/Qwen3.5-4B",
    "labels": LABELS,
    "max_options": 4,
    "temperature": {"choice": 0.75, "noul": 0.8, "score": 0.65},
    "noul_floor": 0.801,
    "prompt": {
        "chat_template": True,
        "enable_thinking": False,
        "system": SYSTEM,
        "layout": "plain",
        "state_compact": True,
        "prefill": "Answer:",
        "max_state_tokens": 64,
        "default_instructions": "Answer the question below.",
    },
    "two_way": {
        "label_rows": "label_rows.lightning",
        "temperature": {"choice": 0.7, "noul": 0.9, "score": 0.6},
        "noul_floor": 0.85,
        "default": "bidirectional",
    },
    "unknown_future_key": {"ignored": True},
}
STATE = "The patron's renewal failed twice and the loan ends Friday."
QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={
            "circulation": "renewals and holds",
            "billing": "fines",
            "reference": "research help",
        },
    ),
    "urgent": NoulQuestion(instructions="Does the patron convey urgency?"),
    "mood": ScoreQuestion(
        instructions="How frustrated is the patron?", criteria=["calm", "annoyed", "angry"]
    ),
}
WORDS = (
    "system user assistant record question options Answer : ) A B C D c true false the statement "
    'holds it does not 0 1 2 a b x y { } " , ticket plan pro Which team should handle this ? '
    "circulation renewals and holds billing fines reference research help Does patron convey "
    "urgency How frustrated is calm annoyed angry The patron's renewal failed twice loan ends Friday . "
    "You answer typed questions about record ' s"
)


def _tokenizer(words: str = WORDS):
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    split = {w for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(words)}
    vocab = {w: i for i, w in enumerate([*SPECIALS, *sorted(split)])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(s, special=True) for s in SPECIALS])
    out = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    out.chat_template = CHAT_TEMPLATE
    return out


def label_groups(tok) -> list[tuple[str, list[int]]]:
    """A, B and D are one token each; C is read from two (`C` and `c`), as a label of the real
    label_rows.lightning may be."""
    ids = {w: tok.convert_tokens_to_ids(w) for w in ["A", "B", "C", "c", "D"]}
    return [("A", [ids["A"]]), ("B", [ids["B"]]), ("C", [ids["C"], ids["c"]]), ("D", [ids["D"]])]


def write_label_rows(path, embedding: torch.Tensor, groups, meta: dict | None = None) -> None:
    """label_rows.lightning as h2o ships it: the safetensors format with `W` fp32 [n, hidden],
    `token_ids` int64 [n] and the label groups in the metadata (`meta` overrides entries)."""
    from safetensors.torch import save_file

    ids = [t for _, g in groups for t in g]
    metadata = {
        "format": h2o.LABEL_ROWS_FORMAT,
        "hidden": str(embedding.shape[1]),
        "softcap": "none",
        "label_order": json.dumps([label for label, _ in groups]),
        "groups": json.dumps([[label, g] for label, g in groups]),
        **(meta or {}),
    }
    save_file(
        {
            "W": embedding[ids].to(torch.float32).contiguous(),
            "token_ids": torch.tensor(ids, dtype=torch.int64),
        },
        str(path),
        metadata=metadata,
    )


def make_checkpoint(path, serve_config: dict | None = None, *, tie_word_embeddings: bool = True):
    """An h2o-lightning checkpoint directory: config.json, model.safetensors, the tokenizer files
    (with chat_template.jinja), serve_config.json and label_rows.lightning. The label rows are the
    model's output embedding rows (its lm_head, or the input embeddings when tied)."""
    tok = _tokenizer()
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        # mlx-lm's Gated DeltaNet kernel spreads a key head over 32 lanes: Dk >= 32.
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        # h2o-lightning-4b's layout, linear 3 : full 1. Two blocks, so a two-way full-attention
        # layer feeds later layers and the last position's state differs between the readings.
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
        max_position_embeddings=512,
        tie_word_embeddings=tie_word_embeddings,
    )
    lm = transformers.Qwen3_5ForCausalLM(cfg)
    lm.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "serve_config.json").write_text(json.dumps(serve_config or SERVE_CONFIG))
    write_label_rows(
        path / "label_rows.lightning", lm.get_output_embeddings().weight.detach(), label_groups(tok)
    )
    return path


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return make_checkpoint(tmp_path_factory.mktemp("h2o"))


@pytest.fixture(scope="module")
def decider_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("decider")
    (path / "strands_decider_config.json").write_text("{}")
    (path / "serve_config.json").write_text("{}")  # a decider checkpoint wins over serve_config
    return path


class _Hub:
    """Records hf_hub_download and snapshot_download calls; serves `files` of one repo."""

    def __init__(self, local, files):
        self.local, self.files = local, set(files)
        self.downloads, self.snapshots = [], []

    def hf_hub_download(self, repo_id, filename, revision=None, **kwargs):
        self.downloads.append((repo_id, filename, revision, kwargs))
        if filename not in self.files:
            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        return str(self.local / filename)

    def snapshot_download(self, repo_id, revision=None, **kwargs):
        self.snapshots.append((repo_id, revision, kwargs))
        return str(self.local)


# ---------------------------------------------------------------- detection and download


def test_an_h2o_folder_is_detected_and_a_decider_folder_is_not(ckpt, decider_dir):
    """A folder with serve_config.json is h2o-lightning; one that also has a decider config is not."""
    assert h2o.resolve(str(ckpt)) == str(ckpt)
    assert h2o.resolve(str(decider_dir)) is None
    assert h2o.is_h2o(str(ckpt)) and not h2o.is_h2o(str(decider_dir))


def test_a_folder_with_no_config_names_the_decider_config(tmp_path):
    """A folder of neither format falls through to the existing loader and its error message."""
    with pytest.raises(FileNotFoundError, match=r"strands_decider_config\.json"):
        infer.load_engine(str(tmp_path), device="cpu")


def test_an_h2o_hub_repo_downloads_only_the_listed_files(ckpt, monkeypatch):
    """An h2o Hub repo is fetched with allow_patterns limited to the serving files, never the
    shim, the adapter or the assets."""
    hub = _Hub(ckpt, {"serve_config.json"})
    monkeypatch.setattr(h2o, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(h2o, "snapshot_download", hub.snapshot_download)
    assert h2o.resolve("h2oai/h2o-lightning-4b") == str(ckpt)
    [(repo, _, kwargs)] = hub.snapshots
    patterns = kwargs["allow_patterns"]
    assert repo == "h2oai/h2o-lightning-4b"
    assert set(patterns) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "serve_config.json",
        "label_rows.lightning",
    }
    for name in [
        "adapter/adapter_model.safetensors",
        "assets/logo.png",
        "h2o_lightning_shim.py",
        "pooled_readout.py",
        "serve.sh",
    ]:
        assert not any(fnmatch.fnmatch(name, p) for p in patterns), name
    assert all(k != "trust_remote_code" for _, _, _, k in hub.downloads)


def test_the_revision_is_none_unless_given(ckpt, monkeypatch):
    """With no revision the Hub calls get None (the default branch); a given one is passed on."""
    hub = _Hub(ckpt, {"serve_config.json"})
    monkeypatch.setattr(h2o, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(h2o, "snapshot_download", hub.snapshot_download)
    h2o.resolve("h2oai/h2o-lightning-4b")
    h2o.resolve("h2oai/h2o-lightning-4b", "e86f8528")
    assert [rev for _, rev, _ in hub.snapshots] == [None, "e86f8528"]
    assert {rev for _, _, rev, _ in hub.downloads[:3]} == {None}
    assert {rev for _, _, rev, _ in hub.downloads[3:]} == {"e86f8528"}


def test_a_decider_hub_repo_is_downloaded_as_before(monkeypatch):
    """A decider Hub repo (no serve_config.json) reaches the existing loader, whose
    snapshot_download gets the same arguments as before."""
    from strands_decider import modeling

    hub = _Hub(None, set())  # no serve_config.json: not h2o
    monkeypatch.setattr(h2o, "hf_hub_download", hub.hf_hub_download)
    calls = []

    def snapshot(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("stop here")

    monkeypatch.setattr(modeling, "snapshot_download", snapshot)
    with pytest.raises(FileNotFoundError, match="stop here"):
        infer.load_engine("org/strands-decider-2B", device="cpu")
    assert calls == [(("org/strands-decider-2B",), {"revision": None})]
    assert [f for _, f, _, _ in hub.downloads] == ["serve_config.json"]


def test_hub_failures_on_an_h2o_repo_name_the_repo_and_the_cause(ckpt, monkeypatch):
    """Once serve_config.json is found, a Hub failure while checking for the decider configs or
    downloading the files is a FileNotFoundError naming the repo and the cause; a failure to
    fetch serve_config.json itself still means "not h2o"."""
    hub = _Hub(ckpt, {"serve_config.json"})
    monkeypatch.setattr(h2o, "snapshot_download", hub.snapshot_download)

    def flaky(repo_id, filename, revision=None, **kwargs):
        if filename == "serve_config.json":
            return str(ckpt / filename)
        raise ConnectionError("503 Service Unavailable")

    monkeypatch.setattr(h2o, "hf_hub_download", flaky)
    with pytest.raises(
        FileNotFoundError, match=r"h2oai/h2o-lightning-4b.*ConnectionError: 503"
    ) as e:
        h2o.resolve("h2oai/h2o-lightning-4b")
    assert isinstance(e.value.__cause__, ConnectionError)

    monkeypatch.setattr(h2o, "hf_hub_download", hub.hf_hub_download)

    def snapshot_fails(repo_id, revision=None, **kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(h2o, "snapshot_download", snapshot_fails)
    with pytest.raises(
        FileNotFoundError,
        match=r"h2oai/h2o-lightning-4b.*downloading.*TimeoutError: read timed out",
    ):
        h2o.resolve("h2oai/h2o-lightning-4b")

    def no_serve_config(repo_id, filename, revision=None, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(h2o, "hf_hub_download", no_serve_config)
    assert h2o.resolve("org/strands-decider-2B") is None


def test_loading_never_trusts_remote_code(ckpt, monkeypatch):
    """No from_pretrained call made while loading an h2o checkpoint passes trust_remote_code."""
    seen = []
    for owner in (
        transformers.AutoTokenizer,
        transformers.AutoConfig,
        transformers.Qwen3_5ForCausalLM,
    ):
        original = owner.from_pretrained

        def spy(*args, _original=original, _owner=owner, **kwargs):
            seen.append((_owner.__name__, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, "from_pretrained", spy)
    h2o.load_h2o_engine(str(ckpt), device="cpu")
    assert {name for name, _ in seen} == {"AutoTokenizer", "AutoConfig", "Qwen3_5ForCausalLM"}
    assert not any(kwargs.get("trust_remote_code") for _, kwargs in seen)


# ---------------------------------------------------------------- prompt, readout, forward


@pytest.fixture(scope="module")
def two_way(ckpt):
    return h2o.load_h2o_engine(str(ckpt), device="cpu")


@pytest.fixture(scope="module")
def one_way(ckpt):
    return h2o.load_h2o_engine(str(ckpt), device="cpu", attention="one-way")


def _expected_probs(logits_by_id, groups, temperature):
    acc = [math.log(sum(math.exp(logits_by_id[t]) for t in g)) for _, g in groups]
    z = math.log(sum(math.exp(a) for a in acc))
    v = [(a - z) / temperature for a in acc]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    return [x / sum(e) for x in e]


def test_the_default_reading_is_two_way(ckpt, two_way):
    """Without an attention argument, both load_h2o_engine and load_engine read two-way."""
    assert two_way.attention == "two-way"
    assert infer.load_engine(str(ckpt), device="cpu").attention == "two-way"


def test_only_the_full_attention_layers_read_both_ways(two_way):
    """Changing the last position moves the earlier positions' output of a full-attention layer
    in two-way, never of a Gated DeltaNet layer; the layer-by-layer causal forward is the model's."""
    torso = two_way.backend.torso
    torch.manual_seed(1)
    hidden = torch.randn(1, 6, 64)
    moved = hidden.clone()
    moved[:, -1] += torch.randn(64)
    with torch.inference_mode():
        for i, kind in enumerate(torso.config.layer_types):
            a = h2o.run_layer(torso, i, hidden, bidirectional=True)[:, :-1]
            b = h2o.run_layer(torso, i, moved, bidirectional=True)[:, :-1]
            changed = not torch.allclose(a, b, atol=1e-6)
            assert changed == (kind == "full_attention"), (i, kind)
            causal_a = h2o.run_layer(torso, i, hidden, bidirectional=False)[:, :-1]
            causal_b = h2o.run_layer(torso, i, moved, bidirectional=False)[:, :-1]
            torch.testing.assert_close(causal_a, causal_b)
        ids = torch.tensor([[5, 9, 11, 3, 20, 7]])
        torch.testing.assert_close(
            h2o.forward_layers(torso, ids, bidirectional=False),
            torso(input_ids=ids).last_hidden_state,
            atol=1e-5,
            rtol=1e-5,
        )


def test_two_way_logits_are_the_final_hidden_state_times_the_label_rows(two_way, one_way):
    """Two-way logits are the final normed hidden state times the label rows, and differ from
    the one-way ones."""
    r = h2o.render_prompt(two_way.tok, two_way.serve_config, STATE, QUESTIONS["team"])
    ids = two_way.tokenize(r)
    token_ids = [t for label in r.labels for t in two_way.label_rows.group(label)]
    with torch.inference_mode():
        hidden = h2o.forward_layers(two_way.backend.torso, torch.tensor([ids]), bidirectional=True)[
            0, -1
        ]
    want = (two_way.label_rows.rows_for(token_ids).double() @ hidden.double()).tolist()
    got = two_way.label_logits(ids, token_ids)
    assert list(got) == token_ids
    assert list(got.values()) == pytest.approx(want, abs=1e-6)
    causal = one_way.label_logits(ids, token_ids)
    assert max(abs(got[t] - causal[t]) for t in token_ids) > 1e-4  # the two readings differ


def test_the_label_rows_softcap_applies_to_two_way_only(two_way, one_way, monkeypatch):
    """The label-row file's softcap caps the two-way logits; one-way reads the causal LM's own
    logits uncapped, as h2o's one-way path reads vLLM's logprobs."""
    import dataclasses

    r = h2o.render_prompt(two_way.tok, two_way.serve_config, STATE, QUESTIONS["team"])
    ids = two_way.tokenize(r)
    token_ids = [t for label in r.labels for t in two_way.label_rows.group(label)]
    raw_two, raw_one = two_way.label_logits(ids, token_ids), one_way.label_logits(ids, token_ids)
    cap = 0.5 * max(abs(x) for x in [*raw_two.values(), *raw_one.values()])  # so the cap bites
    for engine in (two_way, one_way):
        monkeypatch.setattr(
            engine, "label_rows", dataclasses.replace(engine.label_rows, softcap=cap)
        )
    capped_two, capped_one = (
        two_way.label_logits(ids, token_ids),
        one_way.label_logits(ids, token_ids),
    )
    assert list(capped_two.values()) == pytest.approx(
        [cap * math.tanh(x / cap) for x in raw_two.values()]
    )
    assert capped_one == pytest.approx(raw_one)


def test_one_way_reads_the_causal_lm_logits(one_way):
    """One-way probabilities are the causal LM's last-position logits for the label tokens,
    pooled, tempered and softmaxed."""
    response = one_way.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    lm = one_way.backend.lm
    temperatures = one_way.serve_config.temperature
    for name, q in QUESTIONS.items():
        r = h2o.render_prompt(one_way.tok, one_way.serve_config, STATE, q)
        ids = one_way.tok(r.text, add_special_tokens=False)["input_ids"]
        with torch.inference_mode():
            logits = lm(input_ids=torch.tensor([ids])).logits[0, -1].tolist()
        groups = [(label, one_way.label_rows.group(label)) for label in r.labels]
        probs = _expected_probs(
            {t: logits[t] for _, g in groups for t in g}, groups, temperatures[r.kind]
        )
        answer = response.answers[name]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(
                h2o.commit_noul(probs[0], one_way.serve_config.noul_floor), abs=1e-4
            )
        else:
            assert list(answer.probabilities.values()) == pytest.approx(probs, abs=1e-4)


def test_the_prompt_is_rendered_as_serve_config_fixes_it(two_way):
    """The prompt follows serve_config.json: chat template, record/question/options lines,
    compact JSON state, yes/no and score options, and the Answer: prefill."""
    choice = h2o.render_prompt(two_way.tok, two_way.serve_config, STATE, QUESTIONS["team"])
    assert choice.text == (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nrecord: {STATE}\nquestion: Which team should handle this?\noptions:\n"
        "A) circulation: renewals and holds\nB) billing: fines\nC) reference: research help<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
    )
    a, b = choice.state_span
    assert choice.text[a:b] == STATE
    noul = h2o.render_prompt(
        two_way.tok, two_way.serve_config, {"ticket": "x", "plan": "pro"}, QUESTIONS["urgent"]
    )
    assert 'record: {"ticket":"x","plan":"pro"}\n' in noul.text  # compact JSON
    assert "A) true: the statement holds\nB) false: it does not<|im_end|>" in noul.text
    score = h2o.render_prompt(two_way.tok, two_way.serve_config, STATE, QUESTIONS["mood"])
    assert "A) 0: calm\nB) 1: annoyed\nC) 2: angry<|im_end|>" in score.text
    assert (noul.labels, score.labels) == (["A", "B"], ["A", "B", "C"])


def test_temperatures_and_the_yes_no_floor(two_way):
    """Per-type temperatures and the yes/no floor apply as configured; two-way uses the two_way
    block's values and falls back to the one-way ones."""
    lp = [math.log(0.6), math.log(0.4)]
    p = h2o.probabilities(lp, 0.8)
    want = math.exp(math.log(0.6) / 0.8) / (
        math.exp(math.log(0.6) / 0.8) + math.exp(math.log(0.4) / 0.8)
    )
    assert p[0] == pytest.approx(want)
    assert h2o.commit_noul(0.7, 0.801) == pytest.approx(0.801)  # inside the floor: moved to it
    assert h2o.commit_noul(0.3, 0.801) == pytest.approx(0.199)
    assert h2o.commit_noul(0.95, 0.801) == 0.95  # outside the floor: unchanged
    assert h2o.commit_noul(0.7, 0.0) == 0.7  # a floor of 0 is off
    sc = two_way.serve_config
    assert sc.temperatures("two-way") == SERVE_CONFIG["two_way"]["temperature"]
    assert sc.temperatures("one-way") == SERVE_CONFIG["temperature"]
    assert (sc.floor("two-way"), sc.floor("one-way")) == (0.85, 0.801)
    bare = h2o.H2OServeConfig.from_dict({k: v for k, v in SERVE_CONFIG.items() if k != "two_way"})
    assert bare.temperatures("two-way") == bare.temperatures("one-way")
    assert bare.floor("two-way") == 0.801
    rendered = h2o.render_prompt(two_way.tok, sc, STATE, QUESTIONS["urgent"])
    assert h2o.to_answer(rendered, [0.6, 0.4], sc.floor("two-way")).noul == 0.85


def test_a_label_without_a_token_id_is_refused_by_name(ckpt, tmp_path):
    """A label may have several token ids but never none, and no token id may belong to two
    labels; the error names the label."""
    emb = torch.randn(len(_tokenizer()), 64)
    groups = label_groups(_tokenizer())
    write_label_rows(tmp_path / "ok.lightning", emb, groups)
    rows = h2o.LabelRows.read(str(tmp_path / "ok.lightning"))
    assert rows.labels == LABELS and len(rows.group("C")) == 2
    broken = [*groups[:3], ("D", [])]
    write_label_rows(
        tmp_path / "broken.lightning",
        emb,
        groups,
        {"groups": json.dumps([[label, g] for label, g in broken])},
    )
    with pytest.raises(ValueError, match=r"\['D'\] have no token id"):
        h2o.LabelRows.read(str(tmp_path / "broken.lightning"))
    shared = [*groups[:3], ("D", groups[0][1])]
    write_label_rows(
        tmp_path / "shared.lightning",
        emb,
        groups,
        {"groups": json.dumps([[label, g] for label, g in shared])},
    )
    with pytest.raises(ValueError, match="belongs to labels 'A' and 'D'"):
        h2o.LabelRows.read(str(tmp_path / "shared.lightning"))
    with pytest.raises(ValueError, match="in that order"):
        rows.check_labels(["B", "A"])


def test_a_long_state_is_refused_and_the_prompt_is_tokenised_once(two_way, monkeypatch):
    """The whole prompt is tokenised once; a state over max_state_tokens or a prompt over the
    context window is refused, never truncated."""
    request = SystemOneRequest(state=STATE, questions={"team": QUESTIONS["team"]})
    seen = []
    original = two_way.backend.last_hidden

    def spy(ids, *, bidirectional):
        seen.append(ids)
        return original(ids, bidirectional=bidirectional)

    monkeypatch.setattr(two_way.backend, "last_hidden", spy)
    two_way.evaluate(request)
    r = h2o.render_prompt(two_way.tok, two_way.serve_config, STATE, QUESTIONS["team"])
    assert seen == [two_way.tok(r.text, add_special_tokens=False)["input_ids"]]
    with pytest.raises(ValueError, match="max_state_tokens 64"):
        two_way.evaluate(SystemOneRequest(state=STATE * 10, questions={"team": QUESTIONS["team"]}))
    monkeypatch.setattr(two_way, "max_length", 20)
    with pytest.raises(ValueError, match="context window of 20 tokens"):
        two_way.evaluate(request)


# ---------------------------------------------------------------- ask and serve

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from strands_decider import cli, server  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    SystemOneResponse,
    derive_confidence,
    derive_score_confidence,
)

ASK = [
    "--state",
    STATE,
    "--device",
    "cpu",
    "--noul",
    "Is it urgent?",
    "--choice",
    "Which team?=circulation,billing,reference",
    "--score",
    "How angry?=calm,annoyed,angry",
]
HEALTH_KEYS = {
    "status",
    "model",
    "checkpoint",
    "base_model",
    "num_slots",
    "max_length",
    "temperature",
    "device",
    "prefix_cache",
    "vision",
}
BODY = {"state": STATE, "questions": {name: q.model_dump() for name, q in QUESTIONS.items()}}


def _ask(args, monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)
    return CliRunner().invoke(cli.app, ["ask", *args], env={"COLUMNS": "400"})


def test_ask_answers_in_the_decider_shape(ckpt, monkeypatch):
    """`ask` on an h2o checkpoint exits 0 and prints a SystemOneResponse, as JSON and as text."""
    result = _ask([str(ckpt), *ASK, "--json"], monkeypatch)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"model", "answers", "usage"}
    response = SystemOneResponse.model_validate(payload)
    assert [a.type for a in response.answers.values()] == ["noul", "choice", "score"]
    plain = _ask([str(ckpt), *ASK], monkeypatch)
    assert plain.exit_code == 0, plain.output
    assert "choice_0 ->" in plain.output


def test_one_way_is_selected_from_ask_and_create_app(ckpt, monkeypatch):
    """`ask --attention one-way` and `create_app(attention="one-way")` load the one-way reading;
    an unknown value is refused."""
    loaded = []
    original = h2o.load_h2o_engine

    def spy(*args, **kwargs):
        engine = original(*args, **kwargs)
        loaded.append(engine.attention)
        return engine

    monkeypatch.setattr(h2o, "load_h2o_engine", spy)
    assert _ask([str(ckpt), *ASK, "--attention", "one-way"], monkeypatch).exit_code == 0
    assert _ask([str(ckpt), *ASK, "--attention", "sideways"], monkeypatch).exit_code != 0
    assert loaded == ["one-way"]
    server.create_app(str(ckpt), device="cpu", attention="one-way")
    assert server.get_engine().attention == "one-way"


@pytest.fixture(scope="module")
def client(ckpt):
    return TestClient(server.create_app(str(ckpt), device="cpu"))


def test_serve_answers_and_health_keep_the_decider_shape(client, ckpt):
    """/v1/systemone answers validate as SystemOneResponse with the decider's confidence, and
    /health has the same keys as for a decider checkpoint."""
    health = client.get("/health").json()
    assert set(health) == HEALTH_KEYS
    assert (health["base_model"], health["num_slots"], health["max_length"]) == (
        "Qwen/Qwen3.5-4B",
        4,
        512,
    )
    assert health["temperature"] == SERVE_CONFIG["two_way"]["temperature"]
    assert health["model"] == ckpt.name
    r = client.post("/v1/systemone", json=BODY)
    assert r.status_code == 200, r.text
    response = SystemOneResponse.model_validate(
        {k: v for k, v in r.json().items() if k != "latency_ms"}
    )
    team, mood = response.answers["team"], response.answers["mood"]
    assert response.answers["urgent"].type == "noul"
    assert team.confidence == pytest.approx(
        derive_confidence(list(team.probabilities.values())), abs=2e-4
    )
    assert mood.confidence == pytest.approx(
        derive_score_confidence(list(mood.probabilities.values())), abs=2e-4
    )
    assert mood.legend == {"0": "calm", "1": "annoyed", "2": "angry"}


def test_more_options_than_labels_is_a_caller_error(client, ckpt, monkeypatch, two_way):
    """More options than labels is a ValueError, a 422 from serve and a non-zero exit from ask."""
    five = {f"o{i}": "" for i in range(5)}
    with pytest.raises(ValueError, match="5 options but this model has 4 labels"):
        two_way.ask(STATE, {"q": ChoiceQuestion(instructions="Which?", criteria=five)})
    body = {
        "state": STATE,
        "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": five}},
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422 and "4 labels" in r.json()["detail"]
    result = _ask(
        [str(ckpt), "--state", STATE, "--device", "cpu", "--choice", "Which?=a,b,c,d,e"],
        monkeypatch,
    )
    assert result.exit_code != 0


def test_images_are_refused(client, ckpt, monkeypatch, tmp_path, two_way):
    """Images (request images, serve --vision, ask --image) are refused as unsupported."""
    with pytest.raises(ValueError, match="image input is not supported"):
        two_way.evaluate(
            SystemOneRequest(
                state=STATE, questions={"team": QUESTIONS["team"]}, images=["aGVsbG8="]
            )
        )
    r = client.post("/v1/systemone", json={**BODY, "images": ["aGVsbG8="]})
    assert r.status_code == 422 and "image input is not supported" in r.json()["detail"]
    with pytest.raises(ValueError, match="image input is not supported"):
        server.create_app(str(ckpt), device="cpu", vision=True)
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    result = _ask([str(ckpt), *ASK, "--image", str(image)], monkeypatch)
    assert result.exit_code != 0 and "image input is not supported" in result.output


def test_device_mlx_without_mlx_names_the_extra(ckpt, monkeypatch):
    """`--device mlx` without MLX names the strands-decider[mlx] extra, as for a decider checkpoint."""
    monkeypatch.setattr(infer, "mlx_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"strands-decider\[mlx\]"):
        infer.load_engine(str(ckpt), device="mlx")
    with pytest.raises(RuntimeError, match=r"strands-decider\[mlx\]"):
        server.create_app(str(ckpt), device="mlx")


def test_mlx_reads_the_output_rows_torch_reads(monkeypatch):
    """The MLX backend takes tied vs untied output rows from the checkpoint's text config, as
    torch does, and refuses an untied checkpoint whose mlx-lm model has no lm_head. (Runs
    without mlx: only the choice of rows is checked here; tests/test_h2o_mlx.py runs them.)"""
    from types import SimpleNamespace

    from strands_decider import h2o_mlx

    for tied in (True, False):
        cfg = transformers.Qwen3_5TextConfig(tie_word_embeddings=tied)
        assert h2o_mlx.tied_output_embeddings(cfg) is tied
    decoder = SimpleNamespace(embed_tokens=SimpleNamespace(weight=SimpleNamespace(shape=(32, 16))))
    assert h2o_mlx.MLXBackend(decoder, SimpleNamespace(), "p").tied
    untied = h2o_mlx.MLXBackend(decoder, SimpleNamespace(lm_head=object()), "p", tied=False)
    assert not untied.tied
    with pytest.raises(ValueError, match=r"untied output embeddings.*no lm_head"):
        h2o_mlx.MLXBackend(decoder, SimpleNamespace(), "p", tied=False)
