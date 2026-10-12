"""Saving while the embeddings server is away, completing vectors later, reindexing."""

import asyncio

import pytest

from harness.core.consts import MemorySource, SearchMode
from harness.core.errors import EmbedderUnavailableError, MemoryStoreError
from harness.memory.consts import BACKFILL_BATCH
from tests.consts import AGENT_ID
from tests.fakes.embedder import FAKE_EMBED_MODEL, FakeEmbedder
from tests.unit.memory.semantic.conftest import SemEnv

DECISION = "we decided to store vectors in LanceDB"
QUESTION = "which database was picked?"
REWORDED = "LanceDB is the vector store we chose"
OTHER = "the build runs on every push"
NEW_MODEL = "fake-embed-2"
NEAR_DUPLICATE = 0.97


@pytest.mark.acceptance("03-10.11")
async def test_a_save_without_embeddings_works_and_gains_a_vector_later(sem: SemEnv) -> None:
    sem.embedder.same_meaning(DECISION, QUESTION)
    sem.embedder.available = False

    item_id = await sem.save(DECISION)

    saved = await sem.memory.get(item_id)
    assert saved.embed_model is None
    assert await sem.found("LanceDB") == [DECISION]  # words work at once
    assert await sem.found(QUESTION) == []  # meaning does not, yet
    assert await sem.memory.backfill.pending() == 1

    sem.embedder.available = True
    handled = await sem.memory.backfill.drain()

    assert handled == 1
    assert await sem.memory.backfill.pending() == 0
    assert (await sem.memory.get(item_id)).embed_model == FAKE_EMBED_MODEL
    assert await sem.found(QUESTION, mode=SearchMode.VECTOR) == [DECISION]


async def test_the_backfill_stops_while_the_server_is_still_away(sem: SemEnv) -> None:
    sem.embedder.available = False
    await sem.save(DECISION)

    with pytest.raises(EmbedderUnavailableError):
        await sem.memory.backfill.run_once()

    assert await sem.memory.backfill.pending() == 1


async def test_the_backfill_works_in_batches(sem: SemEnv) -> None:
    sem.embedder.available = False
    count = BACKFILL_BATCH + 3
    for number in range(count):
        await sem.save(f"offline fact number {number}")
    sem.embedder.available = True
    sem.embedder.calls.clear()

    handled = await sem.memory.backfill.drain()

    assert handled == count
    assert [len(texts) for texts, _ in sem.embedder.calls] == [BACKFILL_BATCH, 3]


async def test_a_duplicate_saved_offline_is_merged_into_the_older_item(sem: SemEnv) -> None:
    older = await sem.save(DECISION)
    sem.embedder.close_to(REWORDED, DECISION, similarity=NEAR_DUPLICATE)
    sem.embedder.available = False
    newer = await sem.save(REWORDED)
    assert newer != older

    sem.embedder.available = True
    await sem.memory.backfill.drain()

    kept, merged = await sem.memory.get(older), await sem.memory.get(newer)
    assert kept.content == REWORDED
    assert [version.content for version in await sem.memory.history(older)] == [DECISION]
    assert merged.superseded_by == older
    assert merged.valid_until is not None
    assert [item.id for item in await sem.memory.list(AGENT_ID)] == [older]


async def test_an_unrelated_offline_item_stays_its_own(sem: SemEnv) -> None:
    await sem.save(DECISION)
    sem.embedder.available = False
    await sem.save(OTHER)
    sem.embedder.available = True

    await sem.memory.backfill.drain()

    assert len(await sem.memory.list(AGENT_ID)) == 2


async def test_an_updated_item_loses_its_vector_while_the_server_is_away(sem: SemEnv) -> None:
    item_id = await sem.save(DECISION)
    sem.embedder.available = False

    await sem.memory.update(item_id, actor=MemorySource.USER, content=REWORDED)

    assert (await sem.memory.get(item_id)).embed_model is None
    assert await sem.memory.backfill.pending() == 1


async def test_reindex_re_embeds_everything_with_the_new_model(sem: SemEnv) -> None:
    await sem.save(DECISION)
    await sem.save(OTHER)
    newer = FakeEmbedder(NEW_MODEL)
    newer.same_meaning(DECISION, QUESTION)
    sem.embedder = newer
    await sem.reopen()

    assert await sem.found(QUESTION, mode=SearchMode.VECTOR) == []  # old vectors are ignored
    assert await sem.found("LanceDB") == [DECISION]  # words still work

    await sem.memory.reindex(NEW_MODEL)

    models = {item.embed_model for item in await sem.memory.list(AGENT_ID)}
    assert models == {NEW_MODEL}
    assert (await sem.found(QUESTION, mode=SearchMode.VECTOR))[0] == DECISION


async def test_reindex_refuses_a_model_the_server_does_not_run(sem: SemEnv) -> None:
    with pytest.raises(MemoryStoreError):
        await sem.memory.reindex(NEW_MODEL)


async def test_the_background_job_fills_vectors_on_its_own(
    sem: SemEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("harness.memory.semantic.backfill.consts.BACKFILL_INTERVAL_S", 0.01)
    sem.embedder.available = False
    await sem.save(DECISION)

    sem.semantic.start()
    await asyncio.sleep(0.05)
    still_waiting = await sem.memory.backfill.pending()
    sem.embedder.available = True
    for _ in range(100):
        if await sem.memory.backfill.pending() == 0:
            break
        await asyncio.sleep(0.02)

    assert still_waiting == 1
    assert await sem.memory.backfill.pending() == 0
