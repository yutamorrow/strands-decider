"""h2o-lightning's GGUF build through a llama-server (src/strands_decider/h2o_llama.py).

A fake llama-server (`http.server.ThreadingHTTPServer` on 127.0.0.1, a free port) records each
`/completion` body and answers with the reply a test sets: llama.cpp's shape (build b11856), a
broken JSON, a slow reply or an HTTP error. The prompt settings and tokenizer come from the tiny
h2o-lightning checkpoint of tests/test_h2o.py. Nothing is downloaded and no real llama.cpp runs.
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
import unicodedata
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402
from test_h2o import ASK, QUESTIONS, SERVE_CONFIG, STATE, _ask, _Hub, make_checkpoint  # noqa: E402

from strands_decider import h2o, h2o_llama, infer, server  # noqa: E402
from strands_decider.schema import SystemOneRequest, SystemOneResponse  # noqa: E402

IMAGE_SERVE_CONFIG = {
    **SERVE_CONFIG,
    "image": {"max_images": 2, "max_pixels": 4096, "temperature": 0.65},
}
# h2o-lightning-4b v1.3.1's preprocessor_config.json: 16-pixel patches merged 2x2, so an image's
# sides are multiples of 32, and 64 to 16384 of those 32x32 tokens.
PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


class FakeLlamaServer:
    """Records POST bodies; `reply(body)` gives (status, bytes) and may sleep. GET /props gives
    `props`."""

    MARKER = "<__media_t3st__>"

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.reply = self.label_reply
        self.props = {"media_marker": self.MARKER, "modalities": {"vision": True}}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, data):
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/props":
                    self._send(200, json.dumps(fake.props).encode())
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.bodies.append({"path": self.path, **body})
                self._send(*fake.reply(body))

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @staticmethod
    def logprob(token_id: int) -> float:
        """The fake model's logit of a token: fixed, different per token."""
        return -0.5 - 0.37 * (token_id % 7)

    def label_reply(self, body):
        """llama.cpp's /completion reply with post_sampling_probs: the biased (label) tokens'
        softmax, all the mass between them."""
        ids = [t for t, _ in body["logit_bias"]]
        z = sum(math.exp(self.logprob(t)) for t in ids)
        top = [
            {"id": t, "token": str(t), "bytes": [], "prob": math.exp(self.logprob(t)) / z}
            for t in ids
        ]
        reply = {
            "content": "A",
            "completion_probabilities": [
                {"id": ids[0], "token": "A", "bytes": [], "prob": 0.5, "top_probs": top}
            ],
        }
        return 200, json.dumps(reply).encode()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake():
    server_ = FakeLlamaServer()
    yield server_
    server_.close()


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    """The h2o-lightning checkpoint the prompt settings and tokenizer are read from."""
    path = make_checkpoint(tmp_path_factory.mktemp("h2o-gguf-settings"), IMAGE_SERVE_CONFIG)
    (path / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG))
    return path


def _engine(settings, url, **kwargs):
    return h2o_llama.load(str(settings), url=url, **kwargs)


def _png(size=(64, 64), color="red") -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- answers


def test_answers_every_type_from_token_ids_with_every_label_biased(settings, fake):
    """noul, choice and score answers validate as SystemOneResponse; each question sends its
    prompt as token ids, with every label token in logit_bias and enough n_probs."""
    engine = _engine(settings, fake.url)
    response = engine.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    SystemOneResponse.model_validate(response.model_dump())
    assert [a.type for a in response.answers.values()] == ["choice", "noul", "score"]
    assert len(fake.bodies) == len(QUESTIONS)
    for body, (name, q) in zip(fake.bodies, QUESTIONS.items(), strict=True):
        assert body["path"] == "/completion"
        r = h2o.render_prompt(engine.tok, engine.serve_config, STATE, q)
        assert body["prompt"] == engine.tok(r.text, add_special_tokens=False)["input_ids"]
        biased = [t for t, _ in body["logit_bias"]]
        assert biased == [t for label in r.labels for t in engine.group(label)], name
        assert {b for _, b in body["logit_bias"]} == {h2o_llama.LABEL_BIAS}
        assert body["n_predict"] == 1 and body["n_probs"] >= 2 * len(biased)
        assert body["post_sampling_probs"] is True and body["samplers"] == ["temperature"]
        assert (body["temperature"], body["top_k"], body["top_p"]) == (1.0, 0, 1.0)


def test_probabilities_follow_the_types_temperature_and_the_floor(settings, fake):
    """The server's label log-probabilities are pooled, tempered by type (one-way values) and
    floored as for the other builds."""
    engine = _engine(settings, fake.url)
    response = engine.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    sc = engine.serve_config
    for name, q in QUESTIONS.items():
        r = h2o.render_prompt(engine.tok, sc, STATE, q)
        groups = [(label, engine.group(label)) for label in r.labels]
        logits = {t: fake.logprob(t) for _, g in groups for t in g}
        acc = [math.log(sum(math.exp(logits[t]) for t in g)) for _, g in groups]
        z = math.log(sum(math.exp(a) for a in acc))
        probs = h2o.probabilities([a - z for a in acc], sc.temperature[r.kind])
        answer = response.answers[name]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(h2o.commit_noul(probs[0], sc.noul_floor), abs=1e-4)
        else:
            assert list(answer.probabilities.values()) == pytest.approx(probs, abs=1e-4)


def test_the_reading_is_the_hf_builds_one_way_softmax_over_the_label_tokens(settings, fake):
    """The body asks for the plain distribution: one token, every sampler that could reshape it
    off (temperature 1, then `dist`), the same +100 on every label token, post-sampling
    probabilities. From what llama.cpp then returns (the softmax of the biased logits over the
    whole vocabulary, a little mass left outside the labels) the engine gets, up to a constant,
    the labels' own logits, so its answers are the HF build's one-way formula on those logits:
    `probabilities(group_log_probs(logits), T)` and the yes/no floor."""
    raw = {}  # the "model's" logits of the label tokens, deliberately far from normalised

    def reply(body):
        ids = [t for t, _ in body["logit_bias"]]
        for i, t in enumerate(ids):
            raw[t] = 7.3 - 1.9 * i + 0.25 * (t % 5)
        # the vocabulary's other tokens, after the bias, keep 4e-4 of the mass between them
        z = sum(math.exp(raw[t] + h2o_llama.LABEL_BIAS) for t in ids)
        keep = 1.0 - 4e-4
        top = [
            {"id": t, "token": "", "bytes": [], "prob": keep * math.exp(raw[t] + 100.0) / z}
            for t in ids
        ]
        top.append({"id": 0, "token": "", "bytes": [], "prob": 4e-4})
        top.sort(key=lambda e: -e["prob"])
        return 200, json.dumps(
            {"completion_probabilities": [{"id": ids[0], "prob": top[0]["prob"], "top_probs": top}]}
        ).encode()

    fake.reply = reply
    engine = _engine(settings, fake.url)
    response = engine.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    sc = engine.serve_config
    bodies = list(fake.bodies)
    for body, (name, q) in zip(bodies, QUESTIONS.items(), strict=True):
        biased = [t for t, _ in body["logit_bias"]]
        assert {k: body[k] for k in h2o_llama._SAMPLING} == {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "samplers": ["temperature"],
        }
        assert body["logit_bias"] == [[t, 100.0] for t in biased]
        assert (body["n_predict"], body["post_sampling_probs"], body["cache_prompt"]) == (
            1,
            True,
            False,
        )
        assert body["n_probs"] >= len(biased) + 1  # the labels and what is left outside them
        r = h2o.render_prompt(engine.tok, sc, STATE, q)
        groups = [(label, engine.group(label)) for label in r.labels]
        # the HF build's one-way readout on the same logits (H2OEngine.evaluate)
        want = h2o.probabilities(h2o.group_log_probs(raw, groups), sc.temperature[r.kind])
        got = engine.backend.label_logits(r.text, body["prompt"], biased)
        shift = {t: got[t] - raw[t] for t in biased}
        assert max(shift.values()) - min(shift.values()) == pytest.approx(0.0, abs=1e-9)
        answer = response.answers[name]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(h2o.commit_noul(want[0], sc.noul_floor), abs=1e-4)
        else:
            assert list(answer.probabilities.values()) == pytest.approx(want, abs=1e-4)
    # more than 1e-3 outside the labels means the bias was not applied: refused, not renormalised
    fake.reply = lambda body: (
        200,
        json.dumps(
            {
                "completion_probabilities": [
                    {
                        "top_probs": [
                            {"id": t, "prob": 0.99 / len(body["logit_bias"])}
                            for t, _ in body["logit_bias"]
                        ]
                    }
                ]
            }
        ).encode(),
    )
    with pytest.raises(h2o_llama.LlamaServerError, match="outside the label tokens"):
        engine.evaluate(SystemOneRequest(state=STATE, questions={"q": QUESTIONS["urgent"]}))


# ---------------------------------------------------------------- the server's address


def test_the_address_defaults_to_this_machine_and_follows_the_environment(monkeypatch):
    """No URL: $STRANDS_DECIDER_LLAMA_SERVER_URL, else http://127.0.0.1:8080; only http(s)."""
    monkeypatch.delenv(h2o_llama.URL_ENV, raising=False)
    assert h2o_llama.server_url() == "http://127.0.0.1:8080"
    monkeypatch.setenv(h2o_llama.URL_ENV, "http://localhost:9090/")
    assert h2o_llama.server_url() == "http://localhost:9090"
    assert h2o_llama.server_url("http://[::1]:8081") == "http://[::1]:8081"
    for bad in ("file:///etc/passwd", "ftp://127.0.0.1", "http://", "http://h:99999"):
        with pytest.raises(ValueError, match="llama-server URL"):
            h2o_llama.server_url(bad)


def test_a_request_cannot_change_the_address(settings, fake, monkeypatch):
    """Fields of a request that look like an address (`model`, unknown keys) never change where
    the question is sent."""
    monkeypatch.delenv(h2o_llama.URL_ENV, raising=False)
    client = TestClient(
        server.create_app(str(settings), engine="llama-server", llama_server_url=fake.url)
    )
    body = {
        "state": STATE,
        "questions": {"q": QUESTIONS["urgent"].model_dump()},
        "model": "http://203.0.113.9:8080",
        "llama_server_url": "http://203.0.113.9:8080",
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert len(fake.bodies) == 1  # it reached the fake server, the only one it can reach
    assert server.get_engine().backend.url == fake.url


def test_a_remote_address_warns_once_and_this_machine_does_not(settings):
    """A host other than this machine gets one LlamaServerWarning at start-up saying the traffic
    goes elsewhere (unencrypted over http); loopback addresses get none."""
    with pytest.warns(h2o_llama.LlamaServerWarning, match=r"not on this machine.*unencrypted") as w:
        _engine(settings, "http://192.0.2.10:8080")
    assert len(w) == 1
    with pytest.warns(h2o_llama.LlamaServerWarning) as w:
        h2o_llama.server_url("https://gguf.example.com")
    assert "unencrypted" not in str(w[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for url in ("http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080"):
            h2o_llama.server_url(url)


# ---------------------------------------------------------------- images and readings


def test_images_are_checked_shrunk_and_sent_before_the_record(settings, fake):
    """Images go as multimodal_data with the server's media marker (GET /props) for each, before
    the record; an image over max_pixels is shrunk before it is sent; too many images, or a
    request text holding the marker, are refused before any completion; a server without its
    vision projector is a LlamaServerError."""
    engine = _engine(settings, fake.url)
    request = SystemOneRequest(
        state=STATE, questions={"q": QUESTIONS["urgent"]}, images=[_png((256, 128))]
    )
    engine.evaluate(request)
    [body] = fake.bodies
    prompt = body["prompt"]
    assert set(prompt) == {"prompt_string", "multimodal_data"}
    text = prompt["prompt_string"]
    assert text.count(fake.MARKER) == 1
    assert f"user\n{fake.MARKER}record: {STATE}" in text
    from PIL import Image

    sent = Image.open(io.BytesIO(base64.b64decode(prompt["multimodal_data"][0])))
    # shrunk to max_pixels (90x45), then at the HF processor's size for that (sides of 32)
    sizing = engine.backend.sizing
    assert sent.format == "PNG" and sent.size == sizing.size_for(90, 45)
    with pytest.raises(ValueError, match="at most 2"):
        engine.evaluate(request.model_copy(update={"images": [_png()] * 3}))
    with pytest.raises(ValueError, match="media marker"):
        engine.evaluate(request.model_copy(update={"state": f"see {fake.MARKER}"}))
    assert len(fake.bodies) == 1
    fake.props = {"media_marker": fake.MARKER, "modalities": {"vision": False}}
    with pytest.raises(h2o_llama.LlamaServerError, match="--mmproj"):
        _engine(settings, fake.url).evaluate(request)


def test_an_image_request_is_the_hf_builds_prompt_and_pixels(settings, fake):
    """The image goes where the HF build puts it (its placeholder replaced by the server's
    marker, before the record), in a prompt string in NFC as the HF tokenizer normalises it, and
    at the size and with the pixels the HF build's image processor reads: shrunk to max_pixels,
    then transformers' smart_resize of preprocessor_config.json (here up to its 64-token minimum,
    which llama.cpp's own 8-token minimum would not do) with Pillow's bicubic filter. A
    checkpoint without preprocessor_config.json takes no images."""
    from PIL import Image, ImageDraw

    picture = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(picture).ellipse((60, 40, 240, 160), fill="red")
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    image = base64.b64encode(buf.getvalue()).decode()
    state = "Cafe\u0301 receipt"  # "Café" with a combining accent: not NFC
    engine = _engine(settings, fake.url)
    q = QUESTIONS["urgent"]
    engine.evaluate(SystemOneRequest(state=state, questions={"q": q}, images=[image]))
    [body] = fake.bodies

    hf = h2o.render_prompt(engine.tok, engine.serve_config, state, q, n_images=1)
    want = unicodedata.normalize("NFC", hf.text.replace(h2o.IMAGE_PLACEHOLDER, fake.MARKER))
    assert body["prompt"]["prompt_string"] == want
    assert "record: Caf\u00e9 receipt" in want and f"user\n{fake.MARKER}record:" in want

    from strands_decider.h2o_vision import fit_pixels

    fitted = fit_pixels(picture, engine.serve_config.max_pixels)  # what the HF build is given
    assert fitted.size == (78, 52)
    smart = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
    sizing = engine.backend.sizing
    for w, h in ((78, 52), (448, 336), (1182, 886), (4096, 256), (33, 2000)):
        hh, ww = smart.smart_resize(h, w, sizing.factor, sizing.min_pixels, sizing.max_pixels)
        assert sizing.size_for(w, h) == (ww, hh), (w, h)
    sent = Image.open(io.BytesIO(base64.b64decode(body["prompt"]["multimodal_data"][0])))
    assert sent.size == (320, 224)
    expected = fitted.resize((320, 224), Image.Resampling.BICUBIC)
    assert sent.convert("RGB").tobytes() == expected.tobytes()
    if hasattr(transformers, "Qwen2VLImageProcessorPil"):
        processor = transformers.Qwen2VLImageProcessorPil.from_pretrained(str(settings))
        hf_in = processor(images=[fitted], return_tensors="pt")
        server_in = processor(images=[sent.convert("RGB")], return_tensors="pt")
        assert hf_in["image_grid_thw"].tolist() == server_in["image_grid_thw"].tolist()
        assert torch.equal(hf_in["pixel_values"], server_in["pixel_values"])

    bare = h2o_llama.LlamaServerBackend(fake.url)
    with pytest.raises(ValueError, match=r"preprocessor_config\.json"):
        bare.prepare_images([fitted])


def test_environment_proxies_are_not_used(settings, fake, monkeypatch):
    """http_proxy/https_proxy/all_proxy (either case) pointing at an unreachable proxy change
    nothing: the state and the image still reach the server on 127.0.0.1 and an answer
    returns."""
    for name in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
        monkeypatch.setenv(name.upper(), "http://127.0.0.1:9")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    # urlopen's shared opener reads the environment once; drop it so a regression would show.
    monkeypatch.setattr(h2o_llama.urllib.request, "_opener", None)
    engine = _engine(settings, fake.url)
    image = _png()
    response = engine.evaluate(
        SystemOneRequest(state=STATE, questions={"q": QUESTIONS["urgent"]}, images=[image])
    )
    SystemOneResponse.model_validate(response.model_dump())
    assert "q" in response.answers
    [body] = fake.bodies
    assert f"record: {STATE}" in body["prompt"]["prompt_string"]
    assert len(body["prompt"]["multimodal_data"]) == 1


def test_the_gguf_build_reads_one_way_only(settings, fake, monkeypatch):
    """Without attention it reads one-way; two-way is refused; a stopped format is refused with
    its reason."""
    assert _engine(settings, fake.url).attention == "one-way"
    assert (
        infer.load_engine(str(settings), engine="llama-server", llama_server_url=fake.url).attention
        == "one-way"
    )
    with pytest.raises(ValueError, match="one-way only"):
        _engine(settings, fake.url, attention="two-way")
    monkeypatch.setitem(h2o.STOPPED_FORMATS, "llama-server", "no label probabilities")
    with pytest.raises(ValueError, match="no label probabilities"):
        _engine(settings, fake.url)


def test_a_hub_id_downloads_only_the_prompt_settings(settings, fake, monkeypatch):
    """For a Hub id, the weights and label rows are not downloaded."""
    hub = _Hub(settings, {"serve_config.json"})
    monkeypatch.setattr(h2o, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(h2o, "snapshot_download", hub.snapshot_download)
    infer.load_engine("h2oai/h2o-lightning-4b", engine="llama-server", llama_server_url=fake.url)
    [(_, _, kwargs)] = hub.snapshots
    assert "model.safetensors" not in kwargs["allow_patterns"]
    assert "label_rows.lightning" not in kwargs["allow_patterns"]
    assert "serve_config.json" in kwargs["allow_patterns"]


# ---------------------------------------------------------------- failures


def test_failures_name_the_address_and_the_cause(settings, fake):
    """No server, a timeout, an HTTP error, broken JSON and a reply without the label tokens are
    each a LlamaServerError naming the address."""
    request = SystemOneRequest(state=STATE, questions={"q": QUESTIONS["urgent"]})
    down = FakeLlamaServer()
    down_url = down.url
    down.close()
    with pytest.raises(
        h2o_llama.LlamaServerError, match=rf"cannot reach llama-server at {down_url}"
    ):
        _engine(settings, down_url).evaluate(request)

    def slow(body):
        time.sleep(1.0)
        return fake.label_reply(body)

    cases = [
        (slow, "did not answer within 0.2 s"),
        (lambda body: (500, b'{"error": "boom"}'), "HTTP 500"),
        (lambda body: (200, b"{not json"), "invalid JSON"),
        (lambda body: (200, b'{"content": "A"}'), "no completion_probabilities"),
        (
            lambda body: (
                200,
                json.dumps(
                    {"completion_probabilities": [{"top_probs": [{"id": 0, "prob": 1.0}]}]}
                ).encode(),
            ),
            "did not return label token",
        ),
        (
            lambda body: (
                200,
                json.dumps(
                    {"completion_probabilities": [{"top_probs": [{"id": 0, "prob": "x"}]}]}
                ).encode(),
            ),
            "not a number",
        ),
        (
            lambda body: (
                200,
                json.dumps(
                    {
                        "completion_probabilities": [
                            {"top_probs": [{"id": t, "prob": 0.1} for t, _ in body["logit_bias"]]}
                        ]
                    }
                ).encode(),
            ),
            "outside the label tokens",
        ),
    ]
    engine = _engine(settings, fake.url, timeout=0.2)
    for reply, message in cases:
        fake.reply = reply
        with pytest.raises(h2o_llama.LlamaServerError, match=message) as e:
            engine.evaluate(request)
        assert fake.url in str(e.value), message


def test_ask_and_serve_answer_through_the_server_and_report_its_failures(
    settings, fake, monkeypatch
):
    """`ask --engine llama-server` and `serve --engine llama-server` answer through the server; a
    failing server is a non-zero exit from ask and a 502 from serve."""
    args = [str(settings), *ASK, "--engine", "llama-server", "--llama-server-url", fake.url]
    result = _ask([*args, "--json"], monkeypatch)
    assert result.exit_code == 0, result.output
    SystemOneResponse.model_validate(json.loads(result.output))
    client = TestClient(
        server.create_app(str(settings), engine="llama-server", llama_server_url=fake.url)
    )
    health = client.get("/health").json()
    assert health["device"] == "llama-server" and health["model"] == settings.name
    body = {"state": STATE, "questions": {"q": QUESTIONS["team"].model_dump()}}
    assert client.post("/v1/systemone", json=body).status_code == 200
    fake.reply = lambda body: (503, b'{"error": "loading model"}')
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 502 and fake.url in r.json()["detail"]
    result = _ask(args, monkeypatch)
    assert result.exit_code != 0 and "HTTP 503" in result.output
