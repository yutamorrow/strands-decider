"""h2o-lightning checkpoints on MLX must answer as on torch, from the same checkpoint.

The tiny checkpoint of tests/test_h2o.py is read by both backends in both readings. Apple
silicon with the mlx extra only; nothing is downloaded.
"""

from __future__ import annotations

import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("MLX runs on Apple silicon only", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from test_h2o import QUESTIONS, STATE, make_checkpoint  # noqa: E402

from strands_decider import h2o, h2o_mlx  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402


def _probabilities(response):
    return {
        name: [a.noul] if a.type == "noul" else list(a.probabilities.values())
        for name, a in response.answers.items()
    }


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return make_checkpoint(tmp_path_factory.mktemp("h2o-mlx"))


@pytest.fixture(scope="module")
def untied_ckpt(tmp_path_factory):
    return make_checkpoint(tmp_path_factory.mktemp("h2o-mlx-untied"), tie_word_embeddings=False)


@pytest.mark.parametrize("attention", ["two-way", "one-way"])
def test_mlx_reads_the_lm_head_of_an_untied_checkpoint_as_torch_does(untied_ckpt, attention):
    """With untied output embeddings, MLX reads the same lm_head rows as torch, so both readings
    give torch's probabilities."""
    request = SystemOneRequest(state=STATE, questions=QUESTIONS)
    want = _probabilities(
        h2o.load_h2o_engine(str(untied_ckpt), device="cpu", attention=attention).evaluate(request)
    )
    engine = h2o_mlx.load(str(untied_ckpt), attention)
    assert not engine.backend.tied
    got = _probabilities(engine.evaluate(request))
    for name in want:
        assert got[name] == pytest.approx(want[name], abs=1.5e-4), name


@pytest.mark.parametrize("attention", ["two-way", "one-way"])
def test_mlx_answers_as_torch_does(ckpt, attention):
    """MLX gives the same probabilities as torch, in both readings, from the same checkpoint."""
    request = SystemOneRequest(state=STATE, questions=QUESTIONS)
    want = _probabilities(
        h2o.load_h2o_engine(str(ckpt), device="cpu", attention=attention).evaluate(request)
    )
    got = _probabilities(h2o_mlx.load(str(ckpt), attention).evaluate(request))
    assert got.keys() == want.keys()
    for name in want:
        assert got[name] == pytest.approx(want[name], abs=1.5e-4), name  # rounded to 4 places


def test_an_mlx_load_failure_names_the_torch_devices(ckpt, monkeypatch):
    """A failed mlx-lm load is a RuntimeError giving the cause and the torch devices to use."""
    import mlx_lm.utils

    from strands_decider import mlx_engine

    def refuse(*args, **kwargs):
        raise ValueError("Received parameters not in model: model.language_model.layers.0.W")

    monkeypatch.setattr(mlx_lm.utils, "load_model", refuse)
    monkeypatch.setattr(mlx_engine, "load_model", refuse)
    with pytest.raises(RuntimeError, match=r"--device cpu \(or cuda / mps\)") as e:
        h2o_mlx.load(str(ckpt))
    assert "Received parameters not in model" in str(e.value)


# ---------------------------------------------------------------- MLX-quantised builds


@pytest.fixture(scope="module")
def quantised_ckpt(ckpt, tmp_path_factory):
    """The tiny checkpoint quantised to 8 bits by mlx-lm, with serve_config.json and the chat
    template copied and no label_rows.lightning (as the community MLX builds ship)."""
    import shutil

    from mlx_lm.convert import convert

    path = tmp_path_factory.mktemp("h2o-mlx-q8") / "q8"
    convert(str(ckpt), mlx_path=str(path), quantize=True, q_bits=8, q_group_size=32)
    for name in ("serve_config.json", "chat_template.jinja", "tokenizer_config.json"):
        if (ckpt / name).exists():
            shutil.copy(ckpt / name, path / name)
    (path / "label_rows.lightning").unlink(missing_ok=True)
    assert h2o.is_mlx_quantised(str(path))
    return path


def _tokenizer_grouped_torch_one_way(ckpt):
    """torch's one-way reading of the unquantised checkpoint with the labels' token ids taken from
    the tokenizer, as a checkpoint without label rows is read."""
    torch_engine = h2o.load_h2o_engine(str(ckpt), device="cpu", attention="one-way")
    return h2o.H2OEngine(
        serve_config=torch_engine.serve_config,
        label_rows=None,
        tokenizer=torch_engine.tok,
        backend=torch_engine.backend,
        attention="one-way",
        config=h2o.H2OEngineConfig(device="cpu"),
        max_length=torch_engine.max_length,
    )


def test_a_quantised_build_reads_one_way_by_default_and_answers_near_torch(ckpt, quantised_ckpt):
    """Without attention, the quantised build reads one-way and answers; 8-bit weights stay within
    0.05 of torch's one-way probabilities of the unquantised weights."""
    engine = h2o_mlx.load(str(quantised_ckpt))
    assert engine.attention == "one-way" and engine.label_rows is None
    request = SystemOneRequest(state=STATE, questions=QUESTIONS)
    got = _probabilities(engine.evaluate(request))
    want = _probabilities(_tokenizer_grouped_torch_one_way(ckpt).evaluate(request))
    for name in want:
        assert got[name] == pytest.approx(want[name], abs=0.05), name


def test_a_quantised_build_without_label_rows_refuses_two_way(quantised_ckpt):
    """Two-way needs label_rows.lightning; asking for it on a build without one is refused."""
    with pytest.raises(ValueError, match=r"no label_rows\.lightning.*two-way"):
        h2o_mlx.load(str(quantised_ckpt), "two-way")


def test_the_output_rows_of_a_float_checkpoint_are_unchanged(ckpt):
    """Reading output rows through the embedding gives exactly the embedding weight's rows."""
    import mlx.core as mx

    engine = h2o_mlx.load(str(ckpt), "one-way")
    ids = [3, 5, 9]
    want = engine.backend.decoder.embed_tokens.weight[mx.array(ids)].astype(mx.float32)
    got = engine.backend.output_rows(ids)
    assert got.tolist() == pytest.approx(want.tolist(), abs=0)


def test_ask_answers_from_a_quantised_build_on_mlx(quantised_ckpt, monkeypatch):
    """`ask --device mlx` answers from the quantised build (one-way by default)."""
    import json

    from test_h2o import ASK, _ask

    args = [str(quantised_ckpt), *[a if a != "cpu" else "mlx" for a in ASK], "--json"]
    result = _ask(args, monkeypatch)
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.output)) == {"model", "answers", "usage"}
