"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .h2o import H2OEngine, H2OEngineConfig, load_h2o_engine, resolve
from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel, checkpoint_dir
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | H2OEngine | None = None
_engine_lock = threading.Lock()


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion (issue #9) when
    two overlapping threads hit the same device. CUDA and CPU are thread-safe for
    inference, and MLX serialises on its own scheduler."""
    return str(device).startswith("mps")


def get_engine() -> SystemOneEngine | H2OEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def _load_h2o(
    path: str,
    *,
    device: str,
    attention: str | None,
    model_name: str,
    attn_implementation: str | None,
    vision: bool = False,
) -> H2OEngine:
    if device == "mlx":
        from .h2o_mlx import load as load_h2o_mlx

        return load_h2o_mlx(path, attention, H2OEngineConfig(device="mlx", model_name=model_name))
    return load_h2o_engine(
        path,
        device=device,
        attention=attention,
        attn_implementation=attn_implementation,
        model_name=model_name,
        vision=vision,
    )


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    attention: str | None = None,
    revision: str | None = None,
) -> FastAPI:
    """The app for `checkpoint`. An h2o-lightning checkpoint (see h2o.py) is read with
    `attention` ("two-way" or "one-way"; None: the build's default, two-way for the HF build);
    `attention` is not used for a strands-decider checkpoint. `revision` is the Hub revision to
    download (None: the default branch). `vision` keeps the vision tower, on torch devices only."""
    global _engine
    from .h2o import ATTENTION_MODES, MLX_IMAGES_UNSUPPORTED
    from .infer import MLX_EXTRA_MESSAGE, mlx_available

    if attention is not None and attention not in ATTENTION_MODES:
        raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
    app = FastAPI(
        title="strands-decider System One",
        version="0.1.0",
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        strict_window=strict_window,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    if vision and device == "mlx":
        # Checked before anything loads, so it is this error with or without the mlx extra.
        raise ValueError(
            f"--vision runs on torch devices (cuda, mps, cpu), not mlx: {MLX_IMAGES_UNSUPPORTED}"
        )
    if (h2o_path := resolve(checkpoint, revision, refuse_quantised=device != "mlx")) is not None:
        if device == "mlx" and not mlx_available():
            raise RuntimeError(MLX_EXTRA_MESSAGE)
        _engine = _load_h2o(
            h2o_path,
            device=device,
            attention=attention,
            model_name=resolved_name,
            attn_implementation=attn_implementation,
            vision=vision,
        )
    elif vision:
        from .vision import load_vision_engine

        _engine = load_vision_engine(
            checkpoint if revision is None else checkpoint_dir(checkpoint, revision),
            config,
            attn_implementation=attn_implementation,
        )
    else:
        source = checkpoint if revision is None else checkpoint_dir(checkpoint, revision)
        if device == "mlx":
            _engine = load_mlx(source, config)
        else:
            model = StrandsDeciderModel.load(source, attn_implementation=attn_implementation)
            _engine = SystemOneEngine(model, config)

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it.
    @app.get("/health")
    @app.get("/ping")
    def health() -> dict:
        eng = get_engine()
        fields = (
            eng.health_fields()
            if isinstance(eng, H2OEngine)
            else {
                "base_model": eng.model.config.base_model,
                "num_slots": eng.model.config.num_slots,
                "max_length": eng.model.config.max_length,
                "temperature": eng.model.config.temperature,
            }
        )
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            **fields,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
        }

    guard: contextlib.AbstractContextManager
    guard = _engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = get_engine()
        started = time.perf_counter()
        try:
            with guard:
                response = eng.evaluate(request)
        except ValueError as exc:
            # Option count over num_slots, malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    attention: str | None = None,
    revision: str | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
        attention=attention,
        revision=revision,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
