"""``DefaultSemanticMemory``: layers 4 and 5b over LanceDB (03 §6)."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
from pydantic import JsonValue

from harness.config import MemoryConfig
from harness.core.consts import (
    ID_PREFIX_MEMORY,
    Decision,
    ErrorCode,
    HookEvent,
    MemoryKind,
    MemoryScope,
    MemorySource,
    SearchMode,
)
from harness.core.errors import HarnessError, MemoryItemNotFoundError, MemoryStoreError
from harness.core.hook_payloads import BeforeMemoryWritePayload
from harness.core.hook_types import HookInput
from harness.core.records import MemoryItem, MemoryVersion, ScoredItem
from harness.core.types import AgentId, MemoryItemId
from harness.memory import consts
from harness.memory.semantic import filters as sql
from harness.memory.semantic import ranking
from harness.memory.semantic.backfill import Backfill
from harness.memory.semantic.types import ALL_SCOPES, SemanticDeps
from harness.memory.semantic.utils import lexical_terms, lexical_text
from harness.memory.semantic.vectors import Vectors
from harness.memory.semantic.writer import Writer, merged_tags


@dataclass(frozen=True, slots=True)
class _Edit:
    """The fields ``update`` may change besides the content; ``None`` leaves one alone."""

    kind: MemoryKind | None = None
    tags: list[str] | None = None
    pinned: bool | None = None
    observed_at: datetime | None = None


def _parse_changes(changes: dict[str, JsonValue]) -> _Edit:
    """Validate the editable fields of ``update``."""
    unknown = set(changes) - consts.ALLOWED_CHANGES
    if unknown:
        raise MemoryStoreError(consts.MSG_UNKNOWN_CHANGE.format(field=sorted(unknown)[0]))
    kind, tags = changes.get(consts.CHANGE_KIND), changes.get(consts.CHANGE_TAGS)
    pinned, when = changes.get(consts.CHANGE_PINNED), changes.get(consts.CHANGE_OBSERVED_AT)
    return _Edit(
        kind=MemoryKind(kind) if isinstance(kind, str) else None,
        tags=[str(tag) for tag in tags] if isinstance(tags, list) else None,
        pinned=pinned if isinstance(pinned, bool) else None,
        observed_at=datetime.fromisoformat(when) if isinstance(when, str) else None,
    )


class DefaultSemanticMemory:
    """Implements ``SemanticMemory``; every write passes ``before_memory_write`` first."""

    def __init__(self, deps: SemanticDeps) -> None:
        self._deps = deps
        self._vectors = Vectors(deps)
        self._writer = Writer(deps)
        self.backfill = Backfill(deps, self._vectors, self._writer)

    # --- writes ---

    async def remember(
        self,
        agent_id: AgentId | None,
        content: str,
        kind: MemoryKind,
        *,
        tags: Sequence[str] = (),
        source: MemorySource = MemorySource.AGENT,
        source_msg_seq: int | None = None,
        observed_at: datetime | None = None,
    ) -> MemoryItemId:
        """Save a fact; when the owner already holds a near-duplicate, update that one.

        Raises:
            MemoryStoreError: The content is empty, a hook denied the write, or the
                table could not be written.
        """
        scope = MemoryScope.GLOBAL if agent_id is None else MemoryScope.AGENT
        draft = BeforeMemoryWritePayload(
            op=consts.OP_REMEMBER,
            item_id=None,
            content=content,
            kind=kind,
            tags=list(tags),
            scope=scope,
            source=source,
        )
        content, tag_list = await self._gate(agent_id, draft)
        when = observed_at or await self._message_time(agent_id, source_msg_seq)
        vector = await self._vectors.one(content, consts.EMBED_KIND_DOCUMENT)
        async with self._writer.lock:
            twin = await self._vectors.duplicate(agent_id, vector) if vector else None
            if twin is not None:
                await self._writer.rewrite(
                    twin,
                    content,
                    actor=source,
                    observed_at=when,
                    vector=vector,
                    embed_model=self._vectors.model_id,
                    changes={consts.CHANGE_TAGS: merged_tags(twin.tags, tag_list)},
                )
                return MemoryItemId(twin.id)
            return await self._insert(
                agent_id,
                content,
                vector,
                kind=kind,
                tags=tag_list,
                source=source,
                source_msg_seq=source_msg_seq,
                observed_at=when,
            )

    async def update(
        self,
        item_id: MemoryItemId,
        *,
        actor: MemorySource,
        content: str | None = None,
        **changes: JsonValue,
    ) -> None:
        """Correct an item in place (``kind``, ``tags``, ``pinned``, ``observed_at``, content).

        Raises:
            MemoryItemNotFoundError: No such item.
            MemoryStoreError: Unknown field, empty content, or a hook denied the write.
        """
        edit = _parse_changes(changes)
        item = await self.get(item_id)
        draft = BeforeMemoryWritePayload(
            op=consts.OP_UPDATE,
            item_id=item_id,
            content=item.content if content is None else content,
            kind=edit.kind or item.kind,
            tags=item.tags if edit.tags is None else edit.tags,
            scope=item.scope,
            source=actor,
        )
        owner = AgentId(item.agent_id) if item.agent_id else None
        new_content, tag_list = await self._gate(owner, draft)
        changed = new_content != item.content
        fields: dict[str, object] = {}
        if draft.kind is not item.kind:
            fields[consts.CHANGE_KIND] = draft.kind
        if tag_list != item.tags:
            fields[consts.CHANGE_TAGS] = tag_list
        if edit.pinned is not None:
            fields[consts.CHANGE_PINNED] = edit.pinned
        vector = (
            await self._vectors.one(new_content, consts.EMBED_KIND_DOCUMENT) if changed else None
        )
        when = edit.observed_at or (self._deps.clock.now() if changed else item.observed_at)
        async with self._writer.lock:
            await self._writer.rewrite(
                item,
                new_content,
                actor=actor,
                observed_at=when,
                vector=vector,
                embed_model=self._vectors.model_id,
                changes=fields,
            )

    async def supersede(
        self,
        old_id: MemoryItemId,
        new_content: str,
        *,
        actor: MemorySource,
        source_msg_seq: int | None = None,
        observed_at: datetime | None = None,
    ) -> MemoryItemId:
        """Replace a fact that changed; the old item expires and points to the new one.

        Raises:
            MemoryItemNotFoundError: No such item.
            MemoryStoreError: Empty content, or a hook denied the write.
        """
        old = await self.get(old_id)
        draft = BeforeMemoryWritePayload(
            op=consts.OP_SUPERSEDE,
            item_id=old_id,
            content=new_content,
            kind=old.kind,
            tags=old.tags,
            scope=old.scope,
            source=actor,
        )
        agent_id = AgentId(old.agent_id) if old.agent_id else None
        content, tag_list = await self._gate(agent_id, draft)
        when = observed_at or await self._message_time(agent_id, source_msg_seq)
        vector = await self._vectors.one(content, consts.EMBED_KIND_DOCUMENT)
        async with self._writer.lock:
            new_id = await self._insert(
                agent_id,
                content,
                vector,
                kind=old.kind,
                tags=tag_list,
                source=actor,
                source_msg_seq=source_msg_seq,
                observed_at=when,
                pinned=old.pinned,
            )
            await self._writer.expire(old.id, superseded_by=new_id)
        return new_id

    async def forget(self, item_id: MemoryItemId) -> None:
        """Soft delete: the item stops being valid now and stays for history.

        Raises:
            MemoryItemNotFoundError: No such item.
        """
        item = await self.get(item_id)
        async with self._writer.lock:
            await self._writer.expire(item.id)

    async def delete(self, item_id: MemoryItemId) -> None:
        """Physically delete an item and its versions (the user's ``memory.delete``).

        Raises:
            MemoryItemNotFoundError: No such item.
        """
        item = await self.get(item_id)
        async with self._writer.lock:
            await self._deps.tables.remove(sql.with_id(item.id))
            await self._deps.tables.remove_versions(sql.versions_of(item.id))

    async def delete_agent(self, agent_id: AgentId) -> None:
        """Physically delete an agent's items and versions; global items are untouched."""
        async with self._writer.lock:
            await self._deps.tables.remove(sql.of_agent(agent_id))
            await self._deps.tables.remove_versions(sql.of_agent(agent_id))

    # --- reads ---

    async def get(self, item_id: MemoryItemId) -> MemoryItem:
        """Return one item.

        Raises:
            MemoryItemNotFoundError: No such item.
        """
        item = await self._deps.tables.get(item_id)
        if item is None:
            raise MemoryItemNotFoundError(consts.MSG_ITEM_NOT_FOUND.format(item_id=item_id))
        return item

    async def by_ids(self, item_ids: Sequence[MemoryItemId]) -> list[MemoryItem]:
        """Return the items with these ids, in the given order; missing ids are skipped."""
        found = {item.id: item for item in await self._deps.tables.find(sql.with_ids(item_ids))}
        return [found[item_id] for item_id in item_ids if item_id in found]

    async def pinned(self, agent_id: AgentId) -> list[MemoryItem]:
        """Return the valid pinned items the agent sees (its own and global), oldest first."""
        where = sql.all_of(sql.visible(agent_id, ALL_SCOPES), sql.valid(), sql.pinned())
        items = await self._deps.tables.find(where)
        return sorted(items, key=lambda item: item.created_at)

    async def history(self, item_id: MemoryItemId) -> list[MemoryVersion]:
        """Return the previous values of an item, oldest first."""
        return await self._deps.tables.versions(item_id)

    async def reindex(self, embed_model: str) -> None:
        """Re-embed every item with the (new) embeddings model the server now runs.

        Raises:
            MemoryStoreError: The server runs another model than ``embed_model``.
            EmbedderUnavailableError: The server is down.
        """
        await self.backfill.reindex(embed_model)

    async def search(
        self,
        agent_id: AgentId,
        query: str,
        *,
        k: int | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        scopes: Sequence[MemoryScope] = ALL_SCOPES,
        filters: dict[str, JsonValue] | None = None,
        include_expired: bool = False,
        as_of: datetime | None = None,
    ) -> list[ScoredItem]:
        """Search what the agent may see: vectors and BM25, fused with RRF.

        The owner and validity filters run inside both searches; ``filters`` (``kind``,
        ``tags``) apply after the fusion. Without the embeddings server the search is
        lexical only, whatever ``mode`` asks for.
        """
        settings = self._deps.config.section(MemoryConfig)
        where = sql.searchable(agent_id, scopes, include_expired=include_expired, as_of=as_of)
        tables, by_vector, lexical = self._deps.tables, list[MemoryItem](), True
        if mode is not SearchMode.LEXICAL:
            vector = await self._vectors.one(query, consts.EMBED_KIND_QUERY)
            if vector is not None:
                embedded = sql.all_of(where, sql.embedded_with(self._vectors.model_id))
                nearest = await tables.nearest(vector, embedded, consts.SEARCH_CANDIDATES)
                by_vector = [item for item, _ in nearest]
                lexical = mode is SearchMode.HYBRID
        by_text = (
            await tables.matching(lexical_terms(query), where, consts.SEARCH_CANDIDATES)
            if lexical
            else []
        )
        now = self._deps.clock.now()
        fused = ranking.fuse(by_vector, by_text, now=now, recency_weight=settings.recency_weight)
        hits = [hit for hit in fused if ranking.passes(hit.item, filters)]
        hits = hits[: k or settings.search_default_k]
        try:
            await tables.touch([hit.item.id for hit in hits], now)
        except MemoryStoreError as error:
            logger.warning("memory access counters were not updated: {}", error)
        return hits

    # --- helpers ---

    async def _insert(
        self,
        agent_id: AgentId | None,
        content: str,
        vector: Sequence[float] | None,
        *,
        kind: MemoryKind,
        tags: Sequence[str],
        source: MemorySource,
        source_msg_seq: int | None,
        observed_at: datetime,
        pinned: bool = False,
    ) -> MemoryItemId:
        now = self._deps.clock.now()
        item = MemoryItem(
            id=self._deps.ids.new(ID_PREFIX_MEMORY),
            scope=MemoryScope.GLOBAL if agent_id is None else MemoryScope.AGENT,
            agent_id=agent_id,
            kind=kind,
            content=content,
            content_norm=lexical_text(content),
            vector=list(vector) if vector is not None else None,
            embed_model=self._vectors.model_id if vector is not None else None,
            tags=list(tags),
            source=source,
            source_msg_seq=source_msg_seq,
            pinned=pinned,
            observed_at=observed_at,
            created_at=now,
            updated_at=now,
        )
        await self._deps.tables.upsert(item)
        return MemoryItemId(item.id)

    async def _message_time(self, agent_id: AgentId | None, seq: int | None) -> datetime:
        """Return when the source message was said, or now when there is none (03 §6.5)."""
        if agent_id is not None and seq is not None:
            found = await self._deps.history.get_messages(agent_id, from_seq=seq, limit=1)
            if found and found[0].seq == seq:
                return found[0].created_at
        return self._deps.clock.now()

    async def _gate(
        self, agent_id: AgentId | None, draft: BeforeMemoryWritePayload
    ) -> tuple[str, list[str]]:
        """Fire ``before_memory_write``; return the content and tags to store.

        Raises:
            MemoryStoreError: A hook denied the write (or asked for an approval, which
                memory writes cannot wait for), or the content is empty.
        """
        name = consts.GLOBAL_AGENT_NAME
        if agent_id is not None:
            try:
                name = (await self._deps.history.get_agent(agent_id)).name
            except HarnessError:
                name = consts.GLOBAL_AGENT_NAME
        out = await self._deps.hooks.emit(
            HookEvent.BEFORE_MEMORY_WRITE,
            HookInput(
                event=HookEvent.BEFORE_MEMORY_WRITE,
                agent_id=agent_id,
                agent_name=name,
                payload=draft.model_dump(mode="json"),
                timestamp=self._deps.clock.now(),
            ),
        )
        if out.decision in {Decision.DENY, Decision.ASK}:
            reason = "; ".join(out.reasons) or consts.MSG_WRITE_DENIED_NO_REASON
            raise MemoryStoreError(
                consts.MSG_WRITE_DENIED.format(reason=reason), code=ErrorCode.HOOK_DENIED
            )
        content, tags = (
            out.payload.get(consts.PAYLOAD_CONTENT),
            out.payload.get(consts.PAYLOAD_TAGS),
        )
        final = content if isinstance(content, str) else draft.content
        if not final.strip():
            raise MemoryStoreError(consts.MSG_EMPTY_CONTENT)
        return final, [str(tag) for tag in tags] if isinstance(tags, list) else draft.tags

    async def list(
        self, agent_id: AgentId | None, *, include_expired: bool = False
    ) -> list[MemoryItem]:
        """Return one owner's items (``None``: the global ones), newest first."""
        where = sql.owned_by(agent_id)
        if not include_expired:
            where = sql.all_of(where, sql.valid())
        items = await self._deps.tables.find(where)
        return sorted(items, key=lambda item: item.observed_at, reverse=True)
