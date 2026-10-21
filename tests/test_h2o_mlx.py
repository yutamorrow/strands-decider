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
