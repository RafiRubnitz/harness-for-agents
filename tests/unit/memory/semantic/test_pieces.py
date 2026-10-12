"""The pure parts and the edges: fusion, filters, terms, schema version, HTTP client, server."""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import lancedb
import pyarrow as pa
import pytest

from harness.core.consts import ErrorCode, MemoryScope
from harness.core.embedding import EMBED_PATH, HEALTH_PATH, EmbedResponse, HealthResponse
from harness.core.errors import EmbedderUnavailableError, MemoryStoreError
from harness.core.utils.paths import HarnessHome
from harness.memory import EmbedderServer, HttpEmbedder
from harness.memory import consts as memory_consts
from harness.memory.semantic import filters, ranking
from harness.memory.semantic.schema import memories_schema
from harness.memory.semantic.tables import SemanticTables
from harness.memory.semantic.utils import lexical_terms, lexical_text
from tests.consts import AGENT_ID
from tests.fakes.clock import FAKE_CLOCK_START
from tests.fakes.embedder import FAKE_EMBED_MODEL
from tests.fakes.recall import memory_item

DIM = 3
BASE_URL = "http://embedder.test"
VECTOR = [0.1, 0.2, 0.3]
PORT = 8765
EXIT_CODE = 3


# --- fusion ---


def test_rrf_adds_the_reciprocal_ranks_of_both_lists() -> None:
    a, b, c = (memory_item(f"mem_{n}", n) for n in "abc")

    fused = ranking.fuse([a, b], [b, c], now=FAKE_CLOCK_START, recency_weight=0.0)

    by_id = {hit.item.id: hit for hit in fused}
    k = memory_consts.RRF_K
    assert fused[0].item.id == b.id
    assert (by_id[b.id].vector_rank, by_id[b.id].bm25_rank) == (2, 1)
    assert by_id[b.id].score == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert by_id[a.id].score == pytest.approx(1 / (k + 1))
    assert (by_id[c.id].vector_rank, by_id[c.id].bm25_rank) == (None, 2)


def test_recency_lifts_a_fresh_item_by_at_most_the_weight() -> None:
    weight = 0.1
    fresh = memory_item("mem_new", "new", at=FAKE_CLOCK_START)
    old = memory_item("mem_old", "old", at=FAKE_CLOCK_START - timedelta(days=3650))

    plain = ranking.fuse([old], [fresh], now=FAKE_CLOCK_START, recency_weight=0.0)
    lifted = ranking.fuse([old], [fresh], now=FAKE_CLOCK_START, recency_weight=weight)

    assert plain[0].score == plain[1].score
    assert lifted[0].item.id == fresh.id
    assert lifted[0].score == pytest.approx(plain[0].score * (1 + weight))
    assert lifted[1].score == pytest.approx(plain[1].score, rel=1e-6)


def test_recency_halves_every_half_life() -> None:
    age = timedelta(days=memory_consts.RECENCY_HALF_LIFE_DAYS)
    item = memory_item("mem_1", "x", at=FAKE_CLOCK_START - age)

    assert ranking.recency(item, FAKE_CLOCK_START) == pytest.approx(0.5)


# --- terms ---


def test_terms_keep_identifiers_whole_and_also_split_them() -> None:
    terms = lexical_terms("See Config_V2.yaml, (now).")

    assert {"config_v2.yaml", "config_v2", "yaml", "see", "now"} <= set(terms)
    assert all(term == term.lower() for term in terms)


def test_the_indexed_text_and_the_query_use_the_same_terms() -> None:
    text = "והקובץ config_v2.yaml"

    assert lexical_text(text).split() == lexical_terms(text)
    assert "קובצ" in lexical_terms(text)  # prefix stripped, final letter folded


# --- filters ---


def test_the_search_filter_always_limits_the_owner_and_the_validity() -> None:
    both = (MemoryScope.AGENT, MemoryScope.GLOBAL)

    default = filters.searchable(AGENT_ID, both, include_expired=False, as_of=None)
    expired = filters.searchable(AGENT_ID, both, include_expired=True, as_of=None)
    past = filters.searchable(AGENT_ID, both, include_expired=True, as_of=FAKE_CLOCK_START)

    assert all(AGENT_ID in text and "global" in text for text in (default, expired, past))
    assert "valid_until IS NULL" in default
    assert "valid_until" not in expired
    assert FAKE_CLOCK_START.isoformat() in past


def test_no_scope_matches_nothing() -> None:
    assert filters.visible(AGENT_ID, ()) == "false"
    assert filters.with_ids([]) == "false"


def test_a_naive_time_or_an_odd_model_name_is_refused() -> None:
    with pytest.raises(MemoryStoreError):
        filters.valid_at(datetime(2026, 1, 1))
    with pytest.raises(MemoryStoreError):
        filters.embedded_with("model' OR '1'='1")


# --- schema version ---


async def _write_version(path: Path, version: int) -> None:
    db = await lancedb.connect_async(path)
    meta = {memory_consts.SCHEMA_VERSION_KEY: str(version).encode()}
    for name in (memory_consts.TABLE_MEMORIES, memory_consts.TABLE_VERSIONS):
        await db.create_table(name, schema=pa.schema(list(memories_schema()), metadata=meta))


@pytest.mark.parametrize("offset", [1, -1])
async def test_tables_of_another_schema_version_are_not_opened(
    tmp_harness_home: HarnessHome, offset: int
) -> None:
    await _write_version(
        tmp_harness_home.semantic_dir, memory_consts.SEMANTIC_SCHEMA_VERSION + offset
    )

    with pytest.raises(MemoryStoreError) as raised:
        await SemanticTables.open(tmp_harness_home.semantic_dir)

    assert raised.value.code is ErrorCode.MEMORY_STORE_FAILED


# --- HTTP client ---


def _embedder(handler: httpx.MockTransport) -> HttpEmbedder:
    return HttpEmbedder(BASE_URL, model_id=FAKE_EMBED_MODEL, dim=DIM, transport=handler)


def _serving(vectors: Sequence[Sequence[float]], status: str = "ok") -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == HEALTH_PATH:
            body = HealthResponse(status=status, model=FAKE_EMBED_MODEL, dim=DIM)
        else:
            assert request.url.path == EMBED_PATH
            body = EmbedResponse(model=FAKE_EMBED_MODEL, vectors=[list(v) for v in vectors])
        return httpx.Response(200, json=body.model_dump())

    return httpx.MockTransport(handle)


async def test_the_client_returns_one_vector_per_text() -> None:
    embedder = _embedder(_serving([VECTOR, VECTOR]))

    assert await embedder.embed(["a", "b"], "document") == [VECTOR, VECTOR]
    assert await embedder.embed([], "query") == []
    assert await embedder.is_available()
    assert (embedder.model_id, embedder.dim) == (FAKE_EMBED_MODEL, DIM)
    await embedder.close()


@pytest.mark.parametrize(
    "transport",
    [
        _serving([VECTOR]),  # one vector for two texts
        _serving([VECTOR, VECTOR[:-1]]),  # a vector of the wrong length
        httpx.MockTransport(lambda request: httpx.Response(503)),
        httpx.MockTransport(lambda request: httpx.Response(200, text="not json")),
    ],
)
async def test_an_unusable_answer_means_unavailable(transport: httpx.MockTransport) -> None:
    embedder = _embedder(transport)

    with pytest.raises(EmbedderUnavailableError) as raised:
        await embedder.embed(["a", "b"], "document")

    assert raised.value.code is ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE


async def test_a_refused_connection_means_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    embedder = _embedder(httpx.MockTransport(refuse))

    with pytest.raises(EmbedderUnavailableError):
        await embedder.embed(["a"], "document")
    assert not await embedder.is_available()


async def test_a_server_that_is_still_loading_is_not_available() -> None:
    assert not await _embedder(_serving([], status="loading")).is_available()


# --- the managed server process ---


class FakeProcess:
    """A child process that exits as soon as it is waited for, or runs until stopped."""

    def __init__(self, *, exits: bool) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self._done = asyncio.Event()
        if exits:
            self._finish(EXIT_CODE)

    def _finish(self, code: int) -> None:
        self.returncode = code
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode or 0

    def terminate(self) -> None:
        self.terminated = True
        self._finish(0)

    def kill(self) -> None:
        self._finish(1)


def _server(tmp_path: Path, processes: list[FakeProcess], *, exits: bool) -> EmbedderServer:
    async def spawn(args: Sequence[str], log_file: Path) -> FakeProcess:
        assert str(PORT) in args
        assert FAKE_EMBED_MODEL in args
        processes.append(FakeProcess(exits=exits))
        return processes[-1]

    return EmbedderServer(
        port=PORT,
        model=FAKE_EMBED_MODEL,
        log_file=tmp_path / "embedder.log",
        warn=lambda code, message: warned.append(code),
        spawn=spawn,
        restart_delay_s=0.0,
    )


warned: list[ErrorCode] = []


async def test_a_server_that_keeps_exiting_is_restarted_then_given_up(tmp_path: Path) -> None:
    warned.clear()
    processes: list[FakeProcess] = []
    server = _server(tmp_path, processes, exits=True)

    server.start()
    await server.wait()

    assert len(processes) == memory_consts.EMBEDDER_MAX_RESTARTS + 1
    assert warned == [ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE]
    await server.stop()


async def test_stopping_ends_the_running_process(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []
    server = _server(tmp_path, processes, exits=False)

    server.start()
    server.start()  # a second start changes nothing
    await asyncio.sleep(0.01)
    await server.stop()

    assert len(processes) == 1
    assert processes[0].terminated
