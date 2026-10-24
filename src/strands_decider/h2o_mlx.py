"""h2o-lightning checkpoints on Apple silicon through MLX (`--device mlx`).

Only the torso forward moves to MLX: mlx-lm's Qwen3.5 decoder layers are called one by one, so
the full-attention layers can be given no mask (two-way) or the causal mask (one-way) while the
Gated DeltaNet layers run as they always do. The prompt, the readout, the temperatures and the
answer are `h2o.py`'s own code.

This backend is checked against the torch one on a tiny random model (tests/test_h2o_mlx.py, Mac
only) and by hand on the real weights (docs/h2o-lightning.md); CI on Linux does not run it.
Importing this module never imports mlx; `load` does.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from .h2o import (
    ATTENTION_MODES,
    DEFAULT_ATTENTION,
    H2OEngine,
    H2OEngineConfig,
    read_checkpoint,
)

_TORCH_HINT = "use --device cpu (or cuda / mps) to serve it on torch instead"


def tied_output_embeddings(text_cfg: Any) -> bool:
    """Whether the checkpoint's output rows are its input embeddings. Read from the text config
    that torch's `Qwen3_5ForCausalLM` is built from, so both backends read the same rows."""
    return bool(getattr(text_cfg, "tie_word_embeddings", False))


def _failure(path: str, exc: BaseException) -> RuntimeError:
    return RuntimeError(
        f"{path}: the MLX backend could not run this h2o-lightning checkpoint "
        f"({type(exc).__name__}: {exc}); {_TORCH_HINT}"
    )


class MLXBackend:
    """mlx-lm's Qwen3.5 text model, read layer by layer."""

    device = "mlx"
    supports_images = False
    supports_two_way = True

    def __init__(self, decoder: Any, owner: Any, path: str, *, tied: bool = True) -> None:
        self.decoder = decoder
        self.owner = owner
        self.path = path
        # Whether the output rows are the input embeddings: the checkpoint's own
        # text_config.tie_word_embeddings, the setting torch's get_output_embeddings follows.
        self.tied = tied
        if not tied and getattr(owner, "lm_head", None) is None:
            raise ValueError(
                "the checkpoint has untied output embeddings, but mlx-lm's model has no lm_head"
            )
        self.hidden_size = int(decoder.embed_tokens.weight.shape[1])
        # mlx-lm's lazy arrays belong to the thread that made them; one forward at a time.
        self._lock = threading.Lock()

    def check_layers(self, layer_types: list[str]) -> None:
        """mlx-lm decides a layer's kind by position; it must match the checkpoint's layer_types,
        or the two-way mask would land on the wrong layers."""
        linear = [bool(getattr(layer, "is_linear", None)) for layer in self.decoder.layers]
        want = [kind == "linear_attention" for kind in layer_types]
        if linear != want:
            raise ValueError(
                f"mlx-lm's Qwen3.5 layers {linear} do not match the checkpoint's layer_types {layer_types}"
            )

    def last_hidden(self, ids: list[int], *, bidirectional: bool) -> torch.Tensor:
        import mlx.core as mx

        from .mlx_engine import _to_torch

        with self._lock:
            try:
                hidden = self.decoder.embed_tokens(mx.array([ids], dtype=mx.int32))
                causal = "causal" if len(ids) > 1 else None
                for layer in self.decoder.layers:
                    mask = None if layer.is_linear or bidirectional else causal
                    hidden = layer(hidden, mask=mask, cache=None)
                last = self.decoder.norm(hidden)[0, -1].astype(mx.float32)
                mx.eval(last)
            except (MemoryError, RuntimeError) as e:
                raise _failure(self.path, e) from e
            return _to_torch(last)

    def output_rows(self, token_ids: list[int]) -> torch.Tensor:
        import mlx.core as mx

        from .mlx_engine import _to_torch

        weight = self.decoder.embed_tokens.weight if self.tied else self.owner.lm_head.weight
        with self._lock:
            rows = weight[mx.array(token_ids, dtype=mx.int32)].astype(mx.float32)
            mx.eval(rows)
            return _to_torch(rows).reshape(len(token_ids), self.hidden_size)


def load(
    path: str, attention: str | None = DEFAULT_ATTENTION, config: H2OEngineConfig | None = None
) -> H2OEngine:
    """An `H2OEngine` with the torso on MLX for the local h2o-lightning checkpoint `path`
    (`attention` None reads two-way).

    A failure to load or run on MLX (out of memory, weight names mlx-lm does not accept, a layer
    layout other than the one assumed) is a RuntimeError naming the cause and the torch devices;
    it never falls back to torch silently.
    """
    from .infer import MLX_EXTRA_MESSAGE, mlx_available

    if not mlx_available():
        raise RuntimeError(MLX_EXTRA_MESSAGE)
    attention = attention or DEFAULT_ATTENTION
    if attention not in ATTENTION_MODES:
        raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
    serve_config, label_rows, tokenizer, text_cfg, max_length = read_checkpoint(path)
    dtype = str(getattr(text_cfg, "dtype", None) or "bfloat16").removeprefix("torch.")
    try:
        import mlx.core as mx

        from .mlx_engine import DEFAULT_CACHE_LIMIT, _load_torso

        mx.set_cache_limit(DEFAULT_CACHE_LIMIT)
        _lm, decoder, owner, _prefix = _load_torso(Path(path), dtype)
        backend = MLXBackend(decoder, owner, path, tied=tied_output_embeddings(text_cfg))
        backend.check_layers(list(text_cfg.layer_types))
    except (MemoryError, RuntimeError, ValueError, KeyError, AttributeError) as e:
        raise _failure(path, e) from e
    engine_config = config or H2OEngineConfig(device="mlx", model_name=serve_config.model)
    return H2OEngine(
        serve_config=serve_config,
        label_rows=label_rows,
        tokenizer=tokenizer,
        backend=backend,
        attention=attention,
        config=engine_config,
        max_length=max_length,
    )
