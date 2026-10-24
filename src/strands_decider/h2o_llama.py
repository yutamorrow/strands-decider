"""h2o-lightning's GGUF build through a llama.cpp `llama-server` the user runs (`--engine
llama-server`).

strands-decider does not run the GGUF weights itself. It renders each question's prompt from the
h2o-lightning checkpoint's own `serve_config.json` and tokenizer (h2o.py, the same prompt as the
HF build) and sends the token ids to `POST {url}/completion` for one token, with an equal
`logit_bias` on every label token of the question and `post_sampling_probs` under a
temperature-1 sampler and nothing else. The equal bias keeps the labels' logit differences exact
and leaves every other token about e^-100 behind, so the probabilities the server returns in
`completion_probabilities[0].top_probs` are the softmax over the label tokens: the causal LM's
own one-way reading (two-way is refused). They are then tempered and floored as h2o.py does for
the other builds. This is how h2o's own llama.cpp backend (`h2o_lightning_llamacpp.py` in
h2oai/h2o-lightning-4b-gguf) reads them.

With images, the prompt is sent as `{"prompt_string": ..., "multimodal_data": [base64 PNG]}`,
with the server's media marker (GET /props `media_marker`; random per server process unless
`LLAMA_MEDIA_MARKER` pins it) in place of each image, before the record, where the HF build puts
its `<|vision_start|><|image_pad|><|vision_end|>`. llama-server, started with the build's
`--mmproj`, wraps each image in `<|vision_start|>`/`<|vision_end|>` itself (tools/mtmd/mtmd.cpp).
The prompt string is put in Unicode NFC first, as the HF tokenizer's normalizer and h2o's own
llama.cpp backend do: llama-server tokenizes it as sent.

The images are checked and shrunk by strands-decider exactly as for the HF build (`fit_pixels`
to `image.max_pixels`), and then resized here to the size the HF build's image processor gives
them: `smart_resize` with the checkpoint's `preprocessor_config.json` (patch_size x merge_size,
`size.shortest_edge` and `size.longest_edge` as its minimum and maximum pixels), with Pillow's
bicubic filter, which is what `Qwen2VLImageProcessorPil` does. llama.cpp's Qwen-VL preprocessing
(tools/mtmd/clip.cpp, PROJECTOR_TYPE_QWEN3VL; mtmd-image.cpp calc_size_preserved_ratio) is the
same smart_resize, but with its own limits: at least 8 tokens (h2o-lightning's processor: 64)
and at most `--image-max-tokens`. An image already at a size within those limits is copied as it
is (mtmd-image.cpp img_tool::resize), so the server reads the pixels the HF build reads. What is
left: an image whose HF size has more tokens than the server's `--image-max-tokens` (1024 in
h2o's settings; only an image near `image.max_pixels` whose size rounds up past it) is shrunk
again by llama-server, and its answer can differ from the HF build's.

What the GGUF files are: h2oai/h2o-lightning-4b-gguf at its tag v1.2.3 (2504cec) holds files
quantised from h2oai/h2o-lightning-4b at 9030a28 (its artifact_manifest_v3.json), the weights of
h2o's tags v1.2 to v1.2.3, not v1.3.1's. This module reads the GGUF build exactly as the HF build
reads one-way; the answers of that GGUF are those of the earlier weights, and are compared with
the HF build at 9030a28 (evaluation/h2o_lightning/run_checks.sh, C2-C4).

The server's address is fixed when the engine is made (`--llama-server-url`, else
`STRANDS_DECIDER_LLAMA_SERVER_URL`, else `http://127.0.0.1:8080`); nothing in a request can change
it. An address that is not this machine gets one warning at start-up. The server's replies are
treated as untrusted input: anything but the documented shape is a `LlamaServerError`.

API facts are from llama.cpp's server at build b11856, and again at b11854, the build the checks
pin (tools/server/README.md, server-common.cpp, server-context.cpp, server-task.cpp;
common/sampling.cpp: the chain is the logit bias, the listed samplers, then `dist`, whose
probabilities `post_sampling_probs` returns sorted; tools/mtmd/). HTTP uses only the standard
library.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import math
import os
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .h2o import (
    LIGHT_DEFAULT_ATTENTION,
    H2OEngine,
    H2OEngineConfig,
    check_not_stopped,
    read_checkpoint,
)

if TYPE_CHECKING:
    from PIL import Image

URL_ENV = "STRANDS_DECIDER_LLAMA_SERVER_URL"
DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 60.0
# The same bias on every label token: their differences stay exact, everything else drops out.
LABEL_BIAS = 100.0
# log(prob) is floored here, as h2o's llama.cpp backend floors it.
_MIN_PROB = 1e-45
# Probability mass outside the label tokens beyond this means the bias was not applied.
_STRAY_MASS = 1e-3
# A reply larger than this is refused rather than read.
_MAX_REPLY_BYTES = 16 * 1024 * 1024
# Every sampler setting that could reshape the returned distribution, turned off.
_SAMPLING = {
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


class LlamaServerError(RuntimeError):
    """The llama-server could not be reached, timed out, or replied with something unexpected.
    The message names the server's address and the cause. HTTP 502 from `serve`."""


class LlamaServerWarning(UserWarning):
    """The llama-server is not on this machine: states and images are sent to another one."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def server_url(url: str | None = None) -> str:
    """The llama-server's base address: `url`, else $STRANDS_DECIDER_LLAMA_SERVER_URL, else
    http://127.0.0.1:8080. Only http and https, with a host, are accepted. A host that is not
    this machine gets one LlamaServerWarning (saying the traffic is unencrypted over http)."""
    chosen = (url or os.environ.get(URL_ENV) or DEFAULT_URL).strip()
    parts = urllib.parse.urlsplit(chosen)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"llama-server URL must be http(s)://host[:port], got {chosen!r}")
    if parts.query or parts.fragment:
        raise ValueError(f"llama-server URL must not carry a query or fragment: {chosen!r}")
    try:
        parts.port  # noqa: B018 - validates the port
    except ValueError as e:
        raise ValueError(f"llama-server URL has an invalid port: {chosen!r}") from e
    base = chosen.rstrip("/")
    if not _is_loopback(parts.hostname):
        plain = " unencrypted (http://)" if parts.scheme == "http" else ""
        warnings.warn(
            f"llama-server at {base} is not on this machine: every state and image is sent to "
            f"it{plain}",
            LlamaServerWarning,
            stacklevel=2,
        )
    return base


@dataclass
class PreparedLlamaImages:
    """A request's images as llama-server takes them: base64 PNGs, in order, and the server's
    media marker that stands for each of them in the prompt."""

    data: list[str]
    placeholder: str
    # The server expands each image itself, so there is nothing to expand here.
    tokens_per_image: None = None


@dataclass(frozen=True)
class ImageSizing:
    """How the HF build's image processor (`Qwen2VLImageProcessorPil`) sizes an image, from the
    checkpoint's `preprocessor_config.json`: sides a multiple of `factor` (patch_size x
    merge_size), at least `min_pixels` and at most `max_pixels` pixels."""

    factor: int
    min_pixels: int
    max_pixels: int

    @classmethod
    def from_file(cls, path: str) -> ImageSizing:
        """The processor's settings, with transformers' defaults where the file leaves one out
        (patch_size 14, merge_size 2, 56x56 to 28x28x1280 pixels)."""
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        size = cfg.get("size") if isinstance(cfg.get("size"), dict) else {}
        values = {
            "patch_size": cfg.get("patch_size", 14),
            "merge_size": cfg.get("merge_size", 2),
            "min_pixels": size.get("shortest_edge", cfg.get("min_pixels", 56 * 56)),
            "max_pixels": size.get("longest_edge", cfg.get("max_pixels", 28 * 28 * 1280)),
        }
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{path}: {key} must be a positive integer, got {value!r}")
        return cls(
            factor=values["patch_size"] * values["merge_size"],
            min_pixels=values["min_pixels"],
            max_pixels=values["max_pixels"],
        )

    def size_for(self, width: int, height: int) -> tuple[int, int]:
        """(width, height) the processor resizes a width x height image to: transformers'
        `smart_resize` (models/qwen2_vl/image_processing_pil_qwen2_vl.py), with Python's round."""
        f = self.factor
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"image is {width}x{height}; its aspect ratio must be smaller than 200"
            )
        h = round(height / f) * f
        w = round(width / f) * f
        if h * w > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h = max(f, math.floor(height / beta / f) * f)
            w = max(f, math.floor(width / beta / f) * f)
        elif h * w < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h = math.ceil(height * beta / f) * f
            w = math.ceil(width * beta / f) * f
        return w, h


def processor_sized(img: Image.Image, sizing: ImageSizing) -> Image.Image:
    """`img` at the size the HF build's image processor gives it, resized as it resizes (Pillow,
    bicubic), so that llama-server, whose own resize keeps an image already at a valid size, reads
    the same pixels."""
    from PIL import Image as PILImage

    img = img.convert("RGB")
    size = sizing.size_for(*img.size)
    if size == img.size:
        return img
    return img.resize(size, PILImage.Resampling.BICUBIC)


class LlamaServerBackend:
    """Reads label log-probabilities from a llama-server's `/completion` endpoint."""

    device = "llama-server"
    hidden_size = 0
    supports_images = True
    supports_two_way = False

    def __init__(
        self, url: str, timeout: float = DEFAULT_TIMEOUT, *, sizing: ImageSizing | None = None
    ) -> None:
        # Fixed here, once; requests never carry an address.
        self.url = url
        self.timeout = float(timeout)
        # How the HF build sizes images; None: the checkpoint takes none.
        self.sizing = sizing
        self._marker: str | None = None
        self._marker_lock = threading.Lock()
        # No proxy, whatever http_proxy/https_proxy/all_proxy say: states and images go to the
        # configured server and nowhere else.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def media_marker(self) -> str:
        """The server's media marker, from GET /props (once). A server without its vision
        projector (`--mmproj`) takes no images."""
        with self._marker_lock:
            if self._marker is None:
                props = self._call("GET", "/props")
                if not isinstance(props, dict):
                    raise LlamaServerError(f"llama-server at {self.url}: /props is not an object")
                modalities = props.get("modalities")
                if isinstance(modalities, dict) and modalities.get("vision") is False:
                    raise LlamaServerError(
                        f"llama-server at {self.url} reads no images: start it with the build's "
                        "--mmproj (mmproj-lightning-f16.gguf)"
                    )
                marker = props.get("media_marker")
                if not isinstance(marker, str) or not marker.strip():
                    raise LlamaServerError(
                        f"llama-server at {self.url}: /props gives no media_marker"
                    )
                self._marker = marker
            return self._marker

    def prepare_images(self, images: Sequence[Image.Image]) -> PreparedLlamaImages:
        check_not_stopped("llama-server-images")
        if self.sizing is None:
            raise ValueError(
                "this checkpoint has no preprocessor_config.json, so the size the HF build reads "
                "an image at is unknown; it takes no images"
            )
        data = []
        for img in images:
            buf = io.BytesIO()
            processor_sized(img, self.sizing).save(buf, format="PNG")
            data.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return PreparedLlamaImages(data=data, placeholder=self.media_marker())

    def last_hidden(self, ids: list[int], *, bidirectional: bool) -> torch.Tensor:
        raise NotImplementedError("a llama-server returns probabilities, not hidden states")

    def output_rows(self, token_ids: list[int]) -> torch.Tensor:
        raise NotImplementedError("a llama-server returns probabilities, not output rows")

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as reply:
                raw = reply.read(_MAX_REPLY_BYTES + 1)
        except urllib.error.HTTPError as e:
            detail = e.read(500).decode("utf-8", "replace").strip()
            raise LlamaServerError(
                f"llama-server at {self.url} answered HTTP {e.code}: {detail or e.reason}"
            ) from e
        except TimeoutError as e:  # socket.timeout too
            raise LlamaServerError(
                f"llama-server at {self.url} did not answer within {self.timeout:g} s"
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise LlamaServerError(
                    f"llama-server at {self.url} did not answer within {self.timeout:g} s"
                ) from e
            raise LlamaServerError(f"cannot reach llama-server at {self.url}: {e.reason}") from e
        except OSError as e:
            raise LlamaServerError(f"cannot reach llama-server at {self.url}: {e}") from e
        if len(raw) > _MAX_REPLY_BYTES:
            raise LlamaServerError(f"llama-server at {self.url} sent a reply over 16 MiB")
        try:
            return json.loads(raw)
        except ValueError as e:
            raise LlamaServerError(f"llama-server at {self.url} sent invalid JSON: {e}") from e

    def _probabilities(self, reply: Any) -> dict[int, float]:
        """{token id: probability} of the first generated position, checked."""
        bad = f"llama-server at {self.url} sent an unexpected reply"
        if not isinstance(reply, dict):
            raise LlamaServerError(f"{bad}: not a JSON object")
        probs = reply.get("completion_probabilities")
        if not isinstance(probs, list) or not probs or not isinstance(probs[0], dict):
            raise LlamaServerError(f"{bad}: no completion_probabilities")
        top = probs[0].get("top_probs")
        if not isinstance(top, list) or not top:
            raise LlamaServerError(f"{bad}: no top_probs at the answer slot")
        out: dict[int, float] = {}
        for item in top:
            if not isinstance(item, dict):
                raise LlamaServerError(f"{bad}: a top_probs entry is not an object")
            tid, p = item.get("id"), item.get("prob")
            if isinstance(tid, bool) or not isinstance(tid, int):
                raise LlamaServerError(f"{bad}: token id {tid!r} is not an integer")
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise LlamaServerError(f"{bad}: probability {p!r} is not a number")
            if not (math.isfinite(p) and 0.0 <= p <= 1.0):
                raise LlamaServerError(f"{bad}: probability {p!r} is outside [0, 1]")
            out[tid] = float(p)
        return out

    def label_logits(
        self,
        text: str,
        ids: list[int],
        token_ids: list[int],
        *,
        images: PreparedLlamaImages | None = None,
    ) -> dict[int, float]:
        """{token id: log-probability} of each label token at the answer slot of the prompt
        (`ids`; `text` with the images' markers when there are images), normalised over the label
        tokens. Every label token must come back, with (almost) all the mass between them."""
        if images is None:
            prompt: Any = ids
        else:
            if text.count(images.placeholder) != len(images.data):
                raise ValueError(
                    "the request's text holds the llama-server's media marker; remove it"
                )
            prompt = {
                "prompt_string": unicodedata.normalize("NFC", text),
                "multimodal_data": images.data,
            }
        body = {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 2 * len(token_ids),
            "post_sampling_probs": True,
            "logit_bias": [[t, LABEL_BIAS] for t in token_ids],
            **_SAMPLING,
            "cache_prompt": False,
            "return_tokens": False,
            "stream": False,
        }
        probs = self._probabilities(self._call("POST", "/completion", body))
        missing = [t for t in token_ids if t not in probs]
        if missing:
            raise LlamaServerError(
                f"llama-server at {self.url} did not return label token(s) {missing[:5]} after "
                "the label bias; is it serving h2o-lightning's GGUF build?"
            )
        stray = 1.0 - sum(probs[t] for t in token_ids)
        if abs(stray) > _STRAY_MASS:
            raise LlamaServerError(
                f"llama-server at {self.url} put {stray:.4f} of the probability outside the "
                "label tokens: the label bias was not applied"
            )
        return {t: math.log(max(probs[t], _MIN_PROB)) for t in token_ids}


def load(
    path: str,
    *,
    url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    attention: str | None = None,
    model_name: str | None = None,
) -> H2OEngine:
    """An `H2OEngine` answering through a llama-server, with the prompt settings and tokenizer of
    the local h2o-lightning checkpoint `path` (`h2o.resolve(..., files=H2O_CONFIG_FILES)` gives
    it without the weights). One-way only; the labels' token ids come from the tokenizer."""
    check_not_stopped("llama-server")
    attention = attention or LIGHT_DEFAULT_ATTENTION
    if attention != "one-way":
        raise ValueError(
            "the GGUF build through llama-server reads one-way only; two-way is not supported "
            "(use --attention one-way)"
        )
    serve_config, _rows, tokenizer, _text_cfg, max_length = read_checkpoint(
        path, require_label_rows=False
    )
    processor_config = os.path.join(path, "preprocessor_config.json")
    sizing = ImageSizing.from_file(processor_config) if os.path.exists(processor_config) else None
    backend = LlamaServerBackend(server_url(url), timeout, sizing=sizing)
    return H2OEngine(
        serve_config=serve_config,
        label_rows=None,
        tokenizer=tokenizer,
        backend=backend,
        attention=attention,
        config=H2OEngineConfig(device="llama-server", model_name=model_name or serve_config.model),
        max_length=max_length,
    )
