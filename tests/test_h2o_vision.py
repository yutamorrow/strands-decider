"""Image input for h2o-lightning checkpoints on torch (src/strands_decider/h2o_vision.py).

A tiny random-weight multimodal Qwen3.5 (3 Gated DeltaNet + 1 full-attention layer, a 2-layer
ViT) is saved with h2o-lightning's serving files: `serve_config.json` with an `image` block,
`label_rows.lightning`, a byte-level Qwen tokenizer with a Qwen-style chat template, and the
PIL image processor's `preprocessor_config.json`. Images are drawn with Pillow. Nothing is
downloaded.
"""

from __future__ import annotations

import base64
import io
import json
import re

import pytest

transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from test_h2o import CHAT_TEMPLATE, LABELS, QUESTIONS, SERVE_CONFIG, write_label_rows  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from strands_decider import cli, h2o, h2o_vision, infer, server  # noqa: E402
from strands_decider.schema import SystemOneRequest, SystemOneResponse  # noqa: E402
from strands_decider.vision import IMAGE_PAD, VISION_END, VISION_START  # noqa: E402

STATE = "The form on the desk."
IMAGE_SERVE_CONFIG = {
    **SERVE_CONFIG,
    "prompt": {**SERVE_CONFIG["prompt"], "max_state_tokens": 256},
    "image": {"max_images": 2, "max_pixels": 4096, "temperature": 0.65},
    "two_way": {**SERVE_CONFIG["two_way"], "image_temperature": 0.55},
}


def _tokenizer():
    from tokenizers import pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, **{c: i + 1 for i, c in enumerate(alphabet)}}
    tok = transformers.Qwen3_5Tokenizer(vocab=vocab, merges=[])
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                VISION_START,
                IMAGE_PAD,
                VISION_END,
            ]
        }
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


def make_vision_checkpoint(path, serve_config: dict | None = None):
    """A multimodal h2o-lightning checkpoint directory (weights, tokenizer, image processor,
    serve_config.json with an image block, label_rows.lightning)."""
    tok = _tokenizer()
    ids = {t: tok.convert_tokens_to_ids(t) for t in (VISION_START, IMAGE_PAD, VISION_END)}
    cfg = transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "head_dim": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": len(tok),
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": True,
            "tie_word_embeddings": True,
            "mtp_num_hidden_layers": 0,
            "max_position_embeddings": 2048,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
                "mrope_section": [3, 3, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "num_heads": 4,
            "intermediate_size": 128,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
        },
        image_token_id=ids[IMAGE_PAD],
        vision_start_token_id=ids[VISION_START],
        vision_end_token_id=ids[VISION_END],
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    lm = transformers.Qwen3_5ForConditionalGeneration(cfg)
    lm.save_pretrained(path)
    tok.save_pretrained(path)
    transformers.Qwen2VLImageProcessorPil(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 1024, "longest_edge": 16384},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    ).save_pretrained(path)
    (path / "serve_config.json").write_text(json.dumps(serve_config or IMAGE_SERVE_CONFIG))
    groups = [(label, [tok.convert_tokens_to_ids(label)]) for label in LABELS]
    write_label_rows(
        path / "label_rows.lightning", lm.get_output_embeddings().weight.detach(), groups
    )
    return path


@pytest.fixture(scope="module")
def vckpt(tmp_path_factory):
    return make_vision_checkpoint(tmp_path_factory.mktemp("h2o-vision"))


@pytest.fixture(scope="module")
def engines(vckpt):
    return {
        attention: h2o.load_h2o_engine(str(vckpt), device="cpu", attention=attention, vision=True)
        for attention in ("two-way", "one-way")
    }


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _picture(color: str, shape: str = "box", size: tuple[int, int] = (64, 64)) -> str:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    box = (8, 8, size[0] - 8, size[1] - 8)
    (draw.ellipse if shape == "disc" else draw.rectangle)(box, fill=color)
    return _b64(img)


def _probabilities(response):
    return {
        name: [a.noul] if a.type == "noul" else list(a.probabilities.values())
        for name, a in response.answers.items()
    }


def _request(*images: str, questions=None) -> SystemOneRequest:
    return SystemOneRequest(state=STATE, questions=questions or QUESTIONS, images=list(images))


# ---------------------------------------------------------------- prompt, images, temperature


def test_images_open_the_user_turn_before_the_record(engines):
    """One placeholder per image opens the user turn, before `record:`; with no image the prompt
    is exactly the text prompt."""
    eng = engines["two-way"]
    text = h2o.render_prompt(eng.tok, eng.serve_config, STATE, QUESTIONS["team"])
    assert (
        h2o.render_prompt(eng.tok, eng.serve_config, STATE, QUESTIONS["team"], n_images=0) == text
    )
    two = h2o.render_prompt(eng.tok, eng.serve_config, STATE, QUESTIONS["team"], n_images=2)
    placeholder = f"{VISION_START}{IMAGE_PAD}{VISION_END}"
    assert f"<|im_start|>user\n{placeholder}{placeholder}record: {STATE}\n" in two.text
    assert two.text.replace(placeholder * 2, "") == text.text
    a, b = two.state_span
    assert two.text[a:b] == STATE
    expanded = h2o_vision.expand_rendered(two, [3, 5])
    assert expanded.text.count(IMAGE_PAD) == 8
    a, b = expanded.state_span
    assert expanded.text[a:b] == STATE


def test_image_requests_use_the_image_temperature(engines, monkeypatch):
    """From the same label logits, an image request uses two_way.image_temperature (two-way) or
    image.temperature (one-way) for every type; a text request uses the per-type ones."""
    logits = {}
    for attention, eng in engines.items():
        groups = {label: eng.group(label) for label in LABELS}
        fixed = {t: 0.3 * k for k, label in enumerate(LABELS) for t in groups[label]}
        monkeypatch.setattr(eng, "label_logits", lambda ids, token_ids, images=None, f=fixed: f)
        logits[attention] = fixed
        for with_image, want_t in (
            (True, {"two-way": 0.55, "one-way": 0.65}[attention]),
            (False, None),
        ):
            request = _request(_picture("red")) if with_image else _request()
            response = eng.evaluate(request)
            for name, q in QUESTIONS.items():
                r = h2o.render_prompt(eng.tok, eng.serve_config, STATE, q)
                t = want_t or eng.serve_config.temperatures(attention)[r.kind]
                grouped = [(label, groups[label]) for label in r.labels]
                probs = h2o.probabilities(h2o.group_log_probs(fixed, grouped), t)
                answer = response.answers[name]
                if answer.type == "noul":
                    floor = eng.serve_config.floor(attention)
                    assert answer.noul == pytest.approx(h2o.commit_noul(probs[0], floor), abs=1e-4)
                else:
                    assert list(answer.probabilities.values()) == pytest.approx(probs, abs=1e-4)


def test_large_images_are_shrunk_and_too_many_are_refused(engines, monkeypatch):
    """An image over image.max_pixels reaches the processor shrunk to at most max_pixels with its
    aspect ratio; more images than image.max_images is a ValueError before any model work."""
    fitted = h2o_vision.fit_pixels(Image.new("RGB", (300, 100)), 4096)
    assert fitted.size[0] * fitted.size[1] <= 4096 and fitted.size[0] / fitted.size[
        1
    ] == pytest.approx(3, rel=0.05)
    assert h2o_vision.fit_pixels(Image.new("RGB", (40, 30)), 4096).size == (40, 30)  # never up
    eng = engines["one-way"]
    seen = []
    processor = eng.backend.image_processor

    class Spy:
        merge_size = processor.merge_size

        def __call__(self, images, **kwargs):
            seen.extend(img.size for img in images)
            return processor(images=images, **kwargs)

    monkeypatch.setattr(eng.backend, "image_processor", Spy())
    eng.evaluate(_request(_picture("red", size=(256, 128)), questions={"team": QUESTIONS["team"]}))
    [(w, h)] = seen
    assert w * h <= 4096 and w / h == pytest.approx(2, rel=0.05)
    with pytest.raises(ValueError, match="at most 2"):
        eng.evaluate(_request(*[_picture("red")] * 3))


def test_a_checkpoint_without_an_image_block_takes_no_images(tmp_path):
    """serve_config.json without `image` means the checkpoint takes no images: loading the vision
    tower for it is refused."""
    bare = {k: v for k, v in IMAGE_SERVE_CONFIG.items() if k != "image"}
    path = make_vision_checkpoint(tmp_path / "no-image", bare)
    assert not h2o.H2OServeConfig.from_dict(bare).takes_images
    with pytest.raises(ValueError, match="no `image` block"):
        h2o.load_h2o_engine(str(path), device="cpu", vision=True)


# ---------------------------------------------------------------- answers and the forward


@pytest.mark.parametrize("attention", ["two-way", "one-way"])
def test_image_requests_answer_and_the_image_moves_the_answer(engines, attention):
    """Both readings answer an image request in the SystemOneResponse shape, and a different
    image gives different probabilities."""
    eng = engines[attention]
    red = eng.evaluate(_request(_picture("red")))
    blue = eng.evaluate(_request(_picture("blue", "disc")))
    SystemOneResponse.model_validate(red.model_dump())
    assert red.usage.input_tokens > 0 and red.usage.output_tokens == len(QUESTIONS)
    diff = max(
        abs(x - y)
        for name in QUESTIONS
        for x, y in zip(_probabilities(red)[name], _probabilities(blue)[name], strict=True)
    )
    assert diff > 1e-4


def test_a_text_request_on_the_vision_engine_answers_as_the_text_engine(vckpt, engines):
    """The vision engine reads a text request as the text engine (decoder only) does."""
    for attention, eng in engines.items():
        text = h2o.load_h2o_engine(str(vckpt), device="cpu", attention=attention)
        want = _probabilities(text.evaluate(_request()))
        got = _probabilities(eng.evaluate(_request()))
        for name in want:
            assert got[name] == pytest.approx(want[name], abs=2e-4), (attention, name)


def _image_prompt(eng, image: str):
    images = eng.backend.prepare_images(h2o_vision.decode_images([image], 4096))
    r = h2o.render_prompt(eng.tok, eng.serve_config, STATE, QUESTIONS["team"], n_images=1)
    ids = eng.tokenize(h2o_vision.expand_rendered(r, images.tokens_per_image))
    return images, torch.tensor([ids])


def test_the_layer_by_layer_forward_is_the_model_forward(engines):
    """With the image features and M-RoPE positions, the layer-by-layer causal forward gives the
    multimodal model's own last hidden states."""
    eng = engines["one-way"]
    image = _picture("green")
    images, input_ids = _image_prompt(eng, image)
    backend = eng.backend
    batch = backend.image_processor(
        images=h2o_vision.decode_images([image], 4096), return_tensors="pt"
    )
    with torch.inference_mode():
        embeds, positions = backend._embed(input_ids, images)
        ours = h2o.forward_layers(
            backend.torso, bidirectional=False, inputs_embeds=embeds, position_ids=positions
        )
        theirs = backend.mm(
            input_ids=input_ids,
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            mm_token_type_ids=(input_ids == backend.image_token_id).to(torch.int32),
        ).last_hidden_state
        backend.mm.rope_deltas = None
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)
    last = backend.last_hidden(input_ids[0].tolist(), bidirectional=False, images=images)
    torch.testing.assert_close(last, theirs[0, -1], atol=1e-5, rtol=1e-5)


def test_two_way_reaches_back_over_the_image_tokens(engines):
    """Changing the last input moves earlier positions (image tokens included) in two-way and
    leaves them as they are in one-way."""
    eng = engines["two-way"]
    images, input_ids = _image_prompt(eng, _picture("red"))
    backend = eng.backend
    with torch.inference_mode():
        embeds, positions = backend._embed(input_ids, images)
        moved = embeds.clone()
        torch.manual_seed(2)
        moved[:, -1] += torch.randn(embeds.shape[-1])
        first_image = int((input_ids[0] == backend.image_token_id).nonzero()[0])
        for bidirectional in (True, False):
            a, b = (
                h2o.forward_layers(
                    backend.torso,
                    bidirectional=bidirectional,
                    inputs_embeds=e,
                    position_ids=positions,
                )[:, :-1]
                for e in (embeds, moved)
            )
            changed = (a - b).abs().amax(dim=-1)[0]
            if bidirectional:
                assert changed[first_image] > 1e-6
            else:
                assert changed.max() < 1e-6


def test_the_vision_tower_runs_once_per_request(engines):
    """Three questions over one image run the vision tower once."""
    eng = engines["two-way"]
    calls = []
    hook = eng.backend.mm.visual.register_forward_hook(lambda *a: calls.append(1))
    try:
        eng.evaluate(_request(_picture("red")))
    finally:
        hook.remove()
    assert len(QUESTIONS) == 3 and len(calls) == 1


# ---------------------------------------------------------------- ask and serve


def _ask(args, monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)
    return CliRunner().invoke(cli.app, ["ask", *args], env={"COLUMNS": "400"})


ASK = ["--state", STATE, "--device", "cpu", "--noul", "Is it signed?", "--choice", "Which?=a,b"]


def test_ask_and_serve_answer_with_images(vckpt, tmp_path, monkeypatch):
    """`ask --image` exits 0 with a SystemOneResponse; `serve --vision` answers a request with
    `images` with 200."""
    image = tmp_path / "form.png"
    image.write_bytes(base64.b64decode(_picture("red")))
    result = _ask([str(vckpt), *ASK, "--image", str(image), "--json"], monkeypatch)
    assert result.exit_code == 0, result.output
    SystemOneResponse.model_validate(json.loads(result.output))
    client = TestClient(server.create_app(str(vckpt), device="cpu", vision=True))
    assert client.get("/health").json()["vision"] is True
    body = {
        "state": STATE,
        "questions": {name: q.model_dump() for name, q in QUESTIONS.items()},
        "images": [_picture("red")],
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    SystemOneResponse.model_validate({k: v for k, v in r.json().items() if k != "latency_ms"})
    r = client.post("/v1/systemone", json={**body, "images": [_picture("red")] * 3})
    assert r.status_code == 422 and "at most 2" in r.json()["detail"]


def test_images_on_mlx_are_refused_before_mlx_loads(vckpt, tmp_path, monkeypatch):
    """`ask --image --device mlx`, `create_app(vision=True, device="mlx")` and
    `load_engine(vision=True, device="mlx")` refuse images on MLX naming the torch devices, even
    without the mlx extra (never the "needs the mlx extra" error)."""
    monkeypatch.setattr(infer, "mlx_available", lambda: False)
    image = tmp_path / "form.png"
    image.write_bytes(base64.b64decode(_picture("red")))
    result = _ask(
        [str(vckpt), *ASK[:2], "--device", "mlx", "--noul", "x?", "--image", str(image)],
        monkeypatch,
    )
    assert result.exit_code != 0
    assert "torch device" in result.output and "mlx extra" not in result.output
    for load in (
        lambda: server.create_app(str(vckpt), device="mlx", vision=True),
        lambda: infer.load_engine(str(vckpt), device="mlx", vision=True),
    ):
        with pytest.raises(ValueError, match="torch device"):
            load()
