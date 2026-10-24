"""Image input for h2o-lightning checkpoints on torch (`serve --vision`, `ask --image`).

An h2o-lightning checkpoint is a multimodal Qwen3.5: its weights hold the vision tower next to
the text decoder. The text path (h2o.py) loads only the decoder; this module loads the whole
`Qwen3_5ForConditionalGeneration` when images are asked for.

The prompt is h2o.py's, with one `<|vision_start|><|image_pad|><|vision_end|>` per image opening
the user turn, before the record, as h2o's own serving places images. Each `<|image_pad|>` is
expanded to the image's token count before the prompt is tokenised once.

Each image is decoded and checked as `vision.decode_image` does (PNG, JPEG, WEBP or GIF, at most
4096 x 4096 pixels, EXIF rotation, transparency on white), shrunk to at most
`serve_config.json`'s `image.max_pixels` (`fit_pixels`), and run through the checkpoint's
`Qwen2VLImageProcessorPil`. The vision tower runs once per request (`prepare_images`); each
question's prompt then gets those features in place of its image tokens, and the decoder runs
layer by layer as on the text path, with the M-RoPE positions of the image prompt. Two-way and
one-way differ only in the full-attention layers' mask, which covers the image tokens too.

Needs transformers >= 5.18 and Pillow (`pip install "strands-decider[vision]"`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from .h2o import H2OServeConfig, RenderedH2O, forward_layers
from .vision import IMAGE_PAD, decode_image, expand_image_tokens, image_tokens_for_grid

if TYPE_CHECKING:
    from PIL import Image


def fit_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """RGB, scaled down (never up), keeping the aspect ratio, to at most `max_pixels` pixels."""
    from PIL import Image as PILImage

    img = img.convert("RGB")
    w, h = img.size
    if max_pixels <= 0 or w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    nw, nh = max(1, math.floor(w * s)), max(1, math.floor(h * s))
    while nw * nh > max_pixels:  # floor of the scale can still land one row over
        if nw >= nh:
            nw -= 1
        else:
            nh -= 1
    return img.resize((max(1, nw), max(1, nh)), PILImage.Resampling.BICUBIC)


def decode_images(images: Sequence[str], max_pixels: int) -> list[Image.Image]:
    """A request's base64 images, checked and decoded (`vision.decode_image`) and shrunk to at
    most `max_pixels` pixels each."""
    return [fit_pixels(decode_image(b), max_pixels) for b in images]


def expand_rendered(rendered: RenderedH2O, tokens_per_image: Sequence[int]) -> RenderedH2O:
    """`rendered` with each `<|image_pad|>` repeated to its image's token count. The images sit
    before the record, so the record's character range moves by the added characters."""
    text = expand_image_tokens(rendered.text, tokens_per_image)
    shift = len(IMAGE_PAD) * (sum(int(n) for n in tokens_per_image) - len(tokens_per_image))
    a, b = rendered.state_span
    return replace(rendered, text=text, state_span=(a + shift, b + shift))


@dataclass
class PreparedImages:
    """One request's images on the model's device: the processor's pixel values and grids, the
    token count of each image, and the vision tower's features (computed once)."""

    grid_thw: torch.Tensor
    tokens_per_image: list[int]
    features: torch.Tensor


class TorchVisionBackend:
    """A multimodal Qwen3.5 (`Qwen3_5ForConditionalGeneration`) on a torch device."""

    supports_images = True
    supports_two_way = True

    def __init__(self, lm: Any, image_processor: Any, device: str) -> None:
        self.lm = lm
        self.mm = lm.model  # Qwen3_5Model: .visual and .language_model
        self.torso = self.mm.language_model
        self.image_processor = image_processor
        self.device = device
        self.hidden_size = int(self.torso.config.hidden_size)
        self.image_token_id = int(lm.config.image_token_id)

    def prepare_images(self, images: Sequence[Image.Image]) -> PreparedImages:
        """Run the processor and the vision tower once for a request's images."""
        with torch.inference_mode():
            batch = self.image_processor(images=list(images), return_tensors="pt")
            grid = batch["image_grid_thw"].to(self.device)
            pixels = batch["pixel_values"].to(self.device)
            out = self.mm.get_image_features(pixels, grid, return_dict=True)
            features = torch.cat(list(out.pooler_output), dim=0)
        return PreparedImages(
            grid_thw=grid,
            tokens_per_image=image_tokens_for_grid(
                grid.tolist(), int(self.image_processor.merge_size)
            ),
            features=features,
        )

    def _embed(
        self, input_ids: torch.Tensor, images: PreparedImages
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The prompt's input embeddings with the image features in place of its image tokens,
        and its [4, 1, L] positions (text row, then the three M-RoPE rows)."""
        embeds = self.torso.embed_tokens(input_ids)
        is_image = input_ids == self.image_token_id
        n = int(is_image.sum())
        if n != images.features.shape[0]:
            raise ValueError(
                f"the prompt holds {n} image tokens but the images give {images.features.shape[0]}"
            )
        embeds = embeds.masked_scatter(
            is_image.unsqueeze(-1).to(embeds.device), images.features.to(embeds.dtype)
        )
        mrope, _ = self.mm.get_rope_index(
            input_ids, is_image.to(torch.int32), image_grid_thw=images.grid_thw
        )
        text = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1)
        return embeds, torch.cat([text, mrope.to(text.device)], dim=0)

    def last_hidden(
        self, ids: list[int], *, bidirectional: bool, images: PreparedImages | None = None
    ) -> torch.Tensor:
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=self.device)
            if images is None:
                if bidirectional:
                    hidden = forward_layers(self.torso, input_ids, bidirectional=True)
                else:
                    hidden = self.torso(
                        input_ids=input_ids, use_cache=False, return_dict=True
                    ).last_hidden_state
            else:
                embeds, positions = self._embed(input_ids, images)
                if bidirectional:
                    hidden = forward_layers(
                        self.torso,
                        bidirectional=True,
                        inputs_embeds=embeds,
                        position_ids=positions,
                    )
                else:
                    hidden = self.torso(
                        inputs_embeds=embeds,
                        position_ids=positions,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
            out: torch.Tensor = hidden[0, -1].to(device="cpu", dtype=torch.float32)
        return out

    def output_rows(self, token_ids: list[int]) -> torch.Tensor:
        with torch.inference_mode():
            weight = self.lm.get_output_embeddings().weight
            out: torch.Tensor = weight[token_ids].to(device="cpu", dtype=torch.float32)
        return out


def load_vision_backend(
    path: str,
    serve_config: H2OServeConfig,
    *,
    device: str,
    dtype: Any,
    attn_implementation: str,
) -> TorchVisionBackend:
    """The multimodal model and image processor of the local h2o-lightning checkpoint `path`.

    Refused (ValueError) when `serve_config.json` has no `image` block, or the checkpoint is not
    a multimodal Qwen3.5 (no vision tower)."""
    from .vision import _check_transformers

    _check_transformers()
    import transformers

    if not serve_config.takes_images:
        raise ValueError(f"{path}: serve_config.json has no `image` block; it takes no images")
    base_cfg = transformers.AutoConfig.from_pretrained(path)
    if base_cfg.model_type != "qwen3_5" or getattr(base_cfg, "vision_config", None) is None:
        raise ValueError(
            f"{path}: model_type {base_cfg.model_type!r} has no vision tower; image input needs "
            "a multimodal qwen3_5 checkpoint"
        )
    # Pinned to the PIL backend, as vision.VisionEngine is: the torchvision one changes pixels.
    processor = transformers.Qwen2VLImageProcessorPil.from_pretrained(path)
    lm: Any = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=dtype, attn_implementation=attn_implementation
    )
    lm.to(device).eval()
    return TorchVisionBackend(lm, processor, device)
