"""Hybrid search: meaning, exact words, Hebrew, owners, validity, fusion, outages."""

from datetime import timedelta

import pytest

from harness.core.consts import ErrorCode, MemoryKind, MemoryScope, MemorySource, SearchMode
from harness.core.errors import MemoryStoreError
from harness.core.types import AgentId
from harness.memory.semantic import filters
from tests.consts import AGENT_ID, OTHER_AGENT_ID
from tests.unit.memory.semantic.conftest import SemEnv

DECISION = "החלטנו על LanceDB"
QUESTION = "איזה מסד נתונים בחרנו?"
FILE_FACT = "the retry limits moved to config_v2.yaml, see the ops channel"
FILE_NAME = "config_v2.yaml"
HEBREW_FACT = "והקובץ נמצא בתיקיית הבית"
HEBREW_WORD = "קובץ"
SECRET_OF_B = "agent B keeps its token in vault-b"
B_QUERY = "vault-b token"
GLOBAL_FACT = "the user prefers short answers"
OLD_VALUE = "the api listens on port 8080"
NEW_VALUE = "the api listens on port 9090"
PORT_QUERY = "api port"
UNRELATED = "lunch is at noon"
HOUR = 3600.0
DAY = 86_400.0
ALL_MODES = list(SearchMode)


# --- what is found ---


@pytest.mark.acceptance("03-10.6")
async def test_a_question_in_other_words_finds_the_fact_by_meaning(sem: SemEnv) -> None:
    sem.embedder.same_meaning(DECISION, QUESTION)
    await sem.save(DECISION)
    await sem.save(UNRELATED)

    hits = await sem.memory.search(AGENT_ID, QUESTION)

    assert hits[0].item.content == DECISION
    assert hits[0].vector_rank == 1
    assert hits[0].bm25_rank is None  # no shared word: only the vectors found it
    assert await sem.found(QUESTION, mode=SearchMode.LEXICAL) == []


@pytest.mark.acceptance("03-10.7")
async def test_an_exact_identifier_is_found_by_words_when_vectors_miss_it(sem: SemEnv) -> None:
    await sem.save(FILE_FACT)
    await sem.save(UNRELATED)

    hits = await sem.memory.search(AGENT_ID, FILE_NAME)

    assert hits[0].item.content == FILE_FACT
    assert hits[0].bm25_rank == 1
    assert await sem.found(FILE_NAME, mode=SearchMode.LEXICAL) == [FILE_FACT]
    vector_only = await sem.memory.search(AGENT_ID, FILE_NAME, mode=SearchMode.VECTOR)
    assert vector_only[0].item.content != FILE_FACT or vector_only[0].score < hits[0].score


@pytest.mark.acceptance("03-10.8")
async def test_a_hebrew_word_finds_its_prefixed_form(sem: SemEnv) -> None:
    await sem.save(HEBREW_FACT)

    assert await sem.found(HEBREW_WORD, mode=SearchMode.LEXICAL) == [HEBREW_FACT]


@pytest.mark.acceptance("03-10.5")
async def test_a_saved_fact_is_found_after_a_restart(sem: SemEnv) -> None:
    sem.embedder.same_meaning(DECISION, QUESTION)
    await sem.save(DECISION)
    await sem.save(FILE_FACT)

    await sem.reopen()

    assert (await sem.found(QUESTION))[0] == DECISION
    assert (await sem.found(FILE_NAME))[0] == FILE_FACT


async def test_k_limits_the_hits_and_defaults_to_the_config(sem: SemEnv) -> None:
    for number in range(5):
        await sem.save(f"shared topic fact number {number}")
    await sem.set("memory.search_default_k", 3)

    by_default = await sem.memory.search(AGENT_ID, "shared topic")
    explicit = await sem.memory.search(AGENT_ID, "shared topic", k=1)

    assert (len(by_default), len(explicit)) == (3, 1)


async def test_kind_and_tag_filters_apply_after_the_fusion(sem: SemEnv) -> None:
    await sem.memory.remember(AGENT_ID, OLD_VALUE, MemoryKind.FACT, tags=["net"])
    await sem.memory.remember(AGENT_ID, NEW_VALUE, MemoryKind.DECISION, tags=["net", "api"])

    by_kind = await sem.memory.search(
        AGENT_ID, PORT_QUERY, filters={"kind": MemoryKind.DECISION.value}
    )
    by_tags = await sem.memory.search(AGENT_ID, PORT_QUERY, filters={"tags": ["net", "api"]})
    by_none = await sem.memory.search(AGENT_ID, PORT_QUERY, filters={"tags": ["missing"]})

    assert [hit.item.content for hit in by_kind] == [NEW_VALUE]
    assert [hit.item.content for hit in by_tags] == [NEW_VALUE]
    assert by_none == []


async def test_a_fresh_fact_outranks_an_equal_old_one(sem: SemEnv) -> None:
    await sem.save("release notes draft alpha")
    sem.clock.advance(DAY * 90)
    await sem.save("release notes draft beta")

    found = await sem.found("release notes draft", mode=SearchMode.LEXICAL)

    assert found[0].endswith("beta")


async def test_a_hit_is_counted_as_an_access(sem: SemEnv) -> None:
    item_id = await sem.save(FILE_FACT)
    sem.clock.advance(HOUR)

    await sem.found(FILE_NAME)
    await sem.found(FILE_NAME)

    item = await sem.memory.get(item_id)
    assert item.access_count == 2
    assert item.last_accessed_at == sem.clock.now()


# --- who may see what ---


@pytest.mark.acceptance("03-10.9")
@pytest.mark.parametrize("mode", ALL_MODES)
async def test_an_agent_never_gets_another_agents_items(sem: SemEnv, mode: SearchMode) -> None:
    sem.embedder.same_meaning(SECRET_OF_B, B_QUERY)
    await sem.save(SECRET_OF_B, agent_id=OTHER_AGENT_ID)
    await sem.save(GLOBAL_FACT, agent_id=None)

    for options in ({}, {"include_expired": True}, {"as_of": sem.clock.now()}):
        hits = await sem.memory.search(AGENT_ID, B_QUERY, mode=mode, **options)
        assert all(hit.item.agent_id in (None, AGENT_ID) for hit in hits)

    assert await sem.found(B_QUERY, agent_id=OTHER_AGENT_ID, mode=mode) != []
    assert await sem.memory.list(AGENT_ID, include_expired=True) == []
    assert await sem.memory.pinned(AGENT_ID) == []


async def test_global_facts_are_seen_by_every_agent_unless_scoped_out(sem: SemEnv) -> None:
    await sem.save(GLOBAL_FACT, agent_id=None)

    everyone = [
        await sem.found("short answers", agent_id=who) for who in (AGENT_ID, OTHER_AGENT_ID)
    ]
    own_only = await sem.memory.search(AGENT_ID, "short answers", scopes=(MemoryScope.AGENT,))

    assert everyone == [[GLOBAL_FACT], [GLOBAL_FACT]]
    assert own_only == []


@pytest.mark.parametrize("hostile", ["x' OR '1'='1", "agt_1; DROP", "", "a b", "אבג"])
async def test_an_id_that_is_not_an_id_never_reaches_a_filter(sem: SemEnv, hostile: str) -> None:
    await sem.save(SECRET_OF_B, agent_id=OTHER_AGENT_ID)

    with pytest.raises(MemoryStoreError):
        await sem.memory.search(AgentId(hostile), B_QUERY)
    with pytest.raises(MemoryStoreError):
        filters.with_ids([hostile])


# --- validity in time ---


@pytest.mark.acceptance("03-10.3")
async def test_a_superseded_fact_is_hidden_unless_the_past_is_asked_for(sem: SemEnv) -> None:
    old_id = await sem.save(OLD_VALUE)
    before_change = sem.clock.now() + timedelta(seconds=HOUR)
    sem.clock.advance(DAY)
    await sem.memory.supersede(old_id, NEW_VALUE, actor=MemorySource.USER)

    now = await sem.found(PORT_QUERY, mode=SearchMode.LEXICAL)
    with_expired = await sem.found(PORT_QUERY, mode=SearchMode.LEXICAL, include_expired=True)
    back_then = await sem.found(PORT_QUERY, mode=SearchMode.LEXICAL, as_of=before_change)

    assert now == [NEW_VALUE]
    assert set(with_expired) == {OLD_VALUE, NEW_VALUE}
    assert back_then == [OLD_VALUE]


async def test_a_forgotten_fact_is_not_found(sem: SemEnv) -> None:
    await sem.memory.forget(await sem.save(FILE_FACT))

    assert await sem.found(FILE_NAME) == []
    assert await sem.found(FILE_NAME, include_expired=True) == [FILE_FACT]


# --- the embeddings server is away ---


@pytest.mark.parametrize("mode", [SearchMode.HYBRID, SearchMode.VECTOR])
async def test_without_embeddings_the_search_is_lexical_and_says_so_once(
    sem: SemEnv, mode: SearchMode
) -> None:
    await sem.save(FILE_FACT)
    sem.embedder.available = False

    first = await sem.memory.search(AGENT_ID, FILE_NAME, mode=mode)
    await sem.memory.search(AGENT_ID, FILE_NAME, mode=mode)

    assert [hit.item.content for hit in first] == [FILE_FACT]
    assert first[0].vector_rank is None
    assert [code for code, _ in sem.warnings] == [ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE]


async def test_a_slow_embeddings_server_counts_as_away(sem: SemEnv) -> None:
    await sem.save(FILE_FACT)
    await sem.set("memory.embed_timeout_s", 0.05)
    sem.embedder.delay_s = 1.0

    assert await sem.found(FILE_NAME) == [FILE_FACT]
    assert [code for code, _ in sem.warnings] == [ErrorCode.MEMORY_EMBEDDER_UNAVAILABLE]


async def test_a_new_outage_is_reported_again(sem: SemEnv) -> None:
    await sem.save(FILE_FACT)
    for available in (False, True, False):
        sem.embedder.available = available
        await sem.found(FILE_NAME)

    assert len(sem.warnings) == 2


async def test_searching_an_empty_memory_finds_nothing(sem: SemEnv) -> None:
    assert await sem.found(FILE_NAME) == []
    assert await sem.found("?!") == []
