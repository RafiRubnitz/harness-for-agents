"""The embeddings server app with a stand-in encoder, and the client talking to it."""

import asyncio
import threading
from collections.abc import Sequence

import httpx
import pytest

from harness.core.embedding import (
    HEALTH_STATUS_FAILED,
    HEALTH_STATUS_LOADING,
    HEALTH_STATUS_OK,
    EmbedKind,
)
from harness.core.errors import EmbedderUnavailableError
from harness.embedder_server import BgeEncoder, create_app
from harness.embedder_server import consts as server_consts
from harness.memory import HttpEmbedder

MODEL = "stand-in-model"
DIM = 4
BASE_URL = "http://embedder.test"
TEXTS = ["first text", "שלום עולם"]
LOAD_ERROR = "weights are missing"


class StandInEncoder:
    """Encodes a text as its length on every axis; loading can be held or made to fail."""

    def __init__(self, *, fails: bool = False) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.fails = fails
        self.seen: list[tuple[list[str], EmbedKind]] = []

    @property
    def model_id(self) -> str:
        return MODEL

    @property
    def dim(self) -> int:
        return DIM

    def load(self) -> None:
        self.gate.wait()
        if self.fails:
            raise RuntimeError(LOAD_ERROR)

    def encode(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        self.seen.append((list(texts), kind))
        return [[float(len(text))] * DIM for text in texts]


async def _status(client: HttpEmbedder, wanted: str) -> str:
    """Poll ``/health`` until it reports ``wanted`` (the model loads in a thread)."""
    status = ""
    for _ in range(200):
        report = await client.health()
        status = report.status if report else ""
        if status == wanted:
            break
        await asyncio.sleep(0.01)
    return status


async def test_the_server_embeds_once_the_model_is_loaded() -> None:
    encoder = StandInEncoder()
    app = create_app(encoder)
    client = HttpEmbedder(BASE_URL, model_id=MODEL, dim=DIM, transport=httpx.ASGITransport(app=app))

    async with app.router.lifespan_context(app):
        assert await _status(client, HEALTH_STATUS_OK) == HEALTH_STATUS_OK
        vectors = await client.embed(TEXTS, "query")

    assert vectors == [[float(len(text))] * DIM for text in TEXTS]
    assert encoder.seen == [(TEXTS, "query")]
    assert await client.is_available()


async def test_while_the_model_loads_health_answers_and_embed_is_refused() -> None:
    encoder = StandInEncoder()
    encoder.gate.clear()
    app = create_app(encoder)
    client = HttpEmbedder(BASE_URL, model_id=MODEL, dim=DIM, transport=httpx.ASGITransport(app=app))

    async with app.router.lifespan_context(app):
        report = await client.health()
        with pytest.raises(EmbedderUnavailableError):
            await client.embed(TEXTS, "document")
        available = await client.is_available()
        encoder.gate.set()
        loaded = await _status(client, HEALTH_STATUS_OK)

    assert report is not None
    assert (report.status, report.model, report.dim) == (HEALTH_STATUS_LOADING, MODEL, DIM)
    assert not available
    assert loaded == HEALTH_STATUS_OK


async def test_a_model_that_fails_to_load_is_reported_by_health() -> None:
    app = create_app(StandInEncoder(fails=True))
    client = HttpEmbedder(BASE_URL, model_id=MODEL, dim=DIM, transport=httpx.ASGITransport(app=app))

    async with app.router.lifespan_context(app):
        status = await _status(client, HEALTH_STATUS_FAILED)

    assert status == HEALTH_STATUS_FAILED
    assert not await client.is_available()


async def test_a_malformed_request_is_rejected() -> None:
    app = create_app(StandInEncoder())
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url=BASE_URL) as raw,
    ):
        response = await raw.post("/embed", json={"texts": TEXTS, "kind": "poem"})

    assert response.is_client_error


def test_the_real_encoder_needs_the_optional_extra_or_a_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_consts, "FLAG_EMBEDDING_MODULE", "no_such_embedding_package")
    encoder = BgeEncoder()

    with pytest.raises(RuntimeError, match="embedder"):
        encoder.load()
    with pytest.raises(RuntimeError):
        encoder.encode(TEXTS, "document")
    assert (encoder.model_id, encoder.dim) == (
        server_consts.DEFAULT_MODEL,
        server_consts.BGE_M3_DIM,
    )
