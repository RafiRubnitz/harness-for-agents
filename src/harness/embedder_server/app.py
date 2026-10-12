"""The FastAPI app of the embeddings server: ``GET /health`` and ``POST /embed``."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from loguru import logger

from harness.core.embedding import (
    EMBED_PATH,
    HEALTH_PATH,
    HEALTH_STATUS_FAILED,
    HEALTH_STATUS_LOADING,
    HEALTH_STATUS_OK,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)
from harness.embedder_server import consts
from harness.embedder_server.encoder import Encoder


class _State:
    """Whether the model is usable yet."""

    def __init__(self) -> None:
        self.status = HEALTH_STATUS_LOADING


async def _load(encoder: Encoder, state: _State) -> None:
    try:
        await asyncio.to_thread(encoder.load)
    except Exception as error:
        logger.error("embeddings model {} failed to load: {}", encoder.model_id, error)
        state.status = HEALTH_STATUS_FAILED
        return
    state.status = HEALTH_STATUS_OK


def create_app(encoder: Encoder) -> FastAPI:
    """Build the server around ``encoder``.

    The model loads in a worker thread after startup, so ``/health`` answers at once
    (``loading`` first, then ``ok`` or ``failed``) and ``/embed`` returns 503 until then.
    """
    state = _State()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        loading = asyncio.create_task(_load(encoder, state))
        try:
            yield
        finally:
            loading.cancel()
            with suppress(asyncio.CancelledError):
                await loading

    app = FastAPI(title=consts.APP_TITLE, lifespan=lifespan)

    @app.get(HEALTH_PATH)
    async def health() -> HealthResponse:  # pyright: ignore[reportUnusedFunction]
        return HealthResponse(status=state.status, model=encoder.model_id, dim=encoder.dim)

    @app.post(EMBED_PATH)
    async def embed(request: EmbedRequest) -> EmbedResponse:  # pyright: ignore[reportUnusedFunction]
        if state.status != HEALTH_STATUS_OK:
            raise HTTPException(
                status_code=consts.STATUS_UNAVAILABLE,
                detail=consts.MSG_NOT_READY.format(status=state.status),
            )
        vectors = await asyncio.to_thread(encoder.encode, request.texts, request.kind)
        return EmbedResponse(model=encoder.model_id, vectors=vectors)

    return app
