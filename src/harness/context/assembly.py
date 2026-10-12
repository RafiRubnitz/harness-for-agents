"""Assembling one model request from the memory layers, in cache-friendly order (07 §3)."""

from collections.abc import Sequence
from dataclasses import dataclass

from harness.config import ContextConfig
from harness.context import history
from harness.context.consts import (
    BASE_INSTRUCTIONS,
    LINE_SEPARATOR,
    MEMORY_DATE_FORMAT,
    MEMORY_ITEM,
    MSG_PINNED_TRIMMED,
    PINNED_HEADER,
    PROFILE_HEADER,
    SECTION_SEPARATOR,
    SUMMARY_BLOCK,
    TAG_HOOK_CONTEXT,
    TAG_MEMORY,
    TAG_SUMMARY,
    THINKING_BUDGET_TOKENS_DEFAULT,
)
from harness.context.types import Budgets, ContextDeps, budgets_for
from harness.context.utils import (
    blocks_tokens,
    estimate_tokens,
    plain_text,
    tagged,
    to_messages,
    with_first_user_blocks,
    with_last_user_text,
)
from harness.core.consts import ID_PREFIX_REQUEST, ErrorCode, MessageRole
from harness.core.records import (
    AgentRecord,
    ContextReport,
    MemoryItem,
    MessageRecord,
    RunRecord,
    SectionName,
    SectionUsage,
    SummaryRecord,
)
from harness.core.types import (
    AgentId,
    CacheHint,
    ContentBlock,
    MemoryItemId,
    Message,
    ModelProfile,
    ModelRequest,
    RequestMeta,
    TextBlock,
    ThinkingConfig,
    ToolSpec,
)

_EXPLICIT_CACHE = "explicit"


@dataclass(frozen=True, slots=True)
class Assembly:
    """A built request with what the caller needs to judge its size."""

    request: ModelRequest
    report: ContextReport
    budgets: Budgets


@dataclass(frozen=True, slots=True)
class _Part:
    """One counted piece of the request."""

    name: SectionName
    text: str
    items: int
    budget: int | None

    def usage(self) -> SectionUsage:
        return SectionUsage(
            name=self.name,
            tokens=estimate_tokens(self.text),
            items=self.items,
            budget_tokens=self.budget,
        )


@dataclass(frozen=True, slots=True)
class _Layers:
    """Everything read from the stores for one build."""

    agent: AgentRecord
    profile: ModelProfile
    settings: ContextConfig
    budgets: Budgets
    summaries: list[SummaryRecord]
    records: list[MessageRecord]
    run: RunRecord | None
    pinned: list[MemoryItem]
    recalled: list[MemoryItem]


def _memory_line(item: MemoryItem) -> str:
    date = item.observed_at.strftime(MEMORY_DATE_FORMAT)
    return MEMORY_ITEM.format(date=date, content=item.content)


def _within(items: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
    """The leading items whose lines fit in ``budget`` tokens."""
    kept: list[MemoryItem] = []
    used = 0
    for item in items:
        used += estimate_tokens(_memory_line(item))
        if used > budget:
            break
        kept.append(item)
    return kept


def _summary_blocks(summaries: Sequence[SummaryRecord]) -> list[ContentBlock]:
    return [
        TextBlock(
            text=SUMMARY_BLOCK.format(
                tag=TAG_SUMMARY, from_seq=s.from_seq, to_seq=s.to_seq, body=s.text
            )
        )
        for s in summaries
    ]


class Assembler:
    """Reads the layers and puts them in order: stable parts first, changing parts last."""

    def __init__(self, deps: ContextDeps) -> None:
        self._deps = deps
        self._trimmed: dict[AgentId, int] = {}

    # --- reading ---

    async def _read(self, agent_id: AgentId, *, retrieve: bool) -> _Layers:
        deps = self._deps
        agent = await deps.memory.get_agent(agent_id)
        profile = deps.gateway.resolve(agent.model)
        settings = deps.config.section(ContextConfig, agent_id)
        summaries = sorted(await deps.memory.get_summaries(agent_id), key=lambda s: s.from_seq)
        covered = max((s.to_seq for s in summaries), default=-1)
        records = await deps.memory.get_messages(agent_id, from_seq=covered + 1)
        run = await deps.memory.open_run(agent_id)
        pinned = await deps.recall.pinned(agent_id) if deps.recall else []
        if retrieve and run is not None:
            run = await self._retrieve(run, records, pinned, settings.auto_retrieve_k)
        recalled: list[MemoryItem] = []
        if deps.recall and run is not None and run.retrieved_memory_ids:
            recalled = await deps.recall.by_ids(run.retrieved_memory_ids)
        return _Layers(
            agent=agent,
            profile=profile,
            settings=settings,
            budgets=budgets_for(profile, settings),
            summaries=summaries,
            records=records,
            run=run,
            pinned=sorted(pinned, key=lambda item: item.created_at),
            recalled=recalled,
        )

    async def _retrieve(
        self, run: RunRecord, records: Sequence[MessageRecord], pinned: Sequence[MemoryItem], k: int
    ) -> RunRecord:
        """Search the semantic memory once, before the run's first answer (07 §6)."""
        recall = self._deps.recall
        in_run = [record for record in records if record.seq >= run.first_seq]
        answered = any(record.role is MessageRole.ASSISTANT for record in in_run)
        if recall is None or k <= 0 or answered or run.retrieved_memory_ids:
            return run
        asked = [plain_text(r.content) for r in in_run if r.role is MessageRole.USER]
        query = LINE_SEPARATOR.join(text for text in asked if text).strip()
        if not query:
            return run
        skip = {item.id for item in pinned}
        found = await recall.relevant(run.agent_id, query, k=k)
        ids = [
            MemoryItemId(item.id)
            for item in found
            if item.id not in skip and item.valid_until is None
        ]
        if not ids:
            return run
        updated = run.model_copy(update={"retrieved_memory_ids": ids})
        await self._deps.memory.save_run(updated)
        return updated

    # --- the stable head ---

    def _system_parts(self, layers: _Layers) -> list[_Part]:
        """Sections 1-4; the oldest pinned items go when the part is over its budget."""
        agent, budget = layers.agent, layers.budgets.system
        profile = self._deps.config.user_profile().strip()
        prompt = agent.system_prompt.strip()
        base = SECTION_SEPARATOR.join(text for text in (BASE_INSTRUCTIONS, prompt) if text)
        fixed = [
            _Part("system", base, 2 if prompt else 1, budget),
            _Part("profile", f"{PROFILE_HEADER}\n{profile}" if profile else "", 1, budget),
        ]
        used = sum(estimate_tokens(part.text) for part in fixed)
        pinned = list(layers.pinned)
        while pinned and used + estimate_tokens(self._pinned_text(pinned)) > budget:
            pinned.pop(0)
        self._warn_trimmed(agent.id, len(layers.pinned) - len(pinned), budget)
        return [*fixed, _Part("pinned", self._pinned_text(pinned), len(pinned), budget)]

    @staticmethod
    def _pinned_text(items: Sequence[MemoryItem]) -> str:
        if not items:
            return ""
        return LINE_SEPARATOR.join([PINNED_HEADER, *(_memory_line(item) for item in items)])

    def _warn_trimmed(self, agent_id: AgentId, dropped: int, budget: int) -> None:
        """Tell the user once per change, not on every step."""
        if self._trimmed.get(agent_id, 0) == dropped:
            return
        self._trimmed[agent_id] = dropped
        if dropped and self._deps.notify is not None:
            message = MSG_PINNED_TRIMMED.format(dropped=dropped, budget=budget)
            self._deps.notify.warning(agent_id, ErrorCode.CONTEXT_PINNED_TRIMMED, message)

    # --- the conversation ---

    def _memory_part(self, layers: _Layers) -> _Part:
        shown = {item.id for item in layers.pinned}
        fresh = [item for item in layers.recalled if item.id not in shown]
        items = _within(fresh, layers.budgets.memories)
        body = LINE_SEPARATOR.join(_memory_line(item) for item in items)
        text = tagged(TAG_MEMORY, body) if items else ""
        return _Part("memories", text, len(items), layers.budgets.memories)

    def _tail(self, layers: _Layers) -> list[MessageRecord]:
        results = layers.settings.tool_results
        tail = history.shrink_results(layers.records, results.truncate_tokens)
        tail = history.stub_old_results(tail, results.keep_recent_steps)
        return history.strip_old_thinking(tail)

    def _messages(
        self, layers: _Layers, tail: Sequence[MessageRecord], memory: _Part, hook: _Part
    ) -> list[Message]:
        head = _summary_blocks(layers.summaries)
        records, pinned = list(tail), False
        if memory.text and layers.run is not None:
            records, pinned = history.pin_text(records, layers.run.first_seq, memory.text)
        if memory.text and not pinned:
            # the message that opened the run was compacted away: keep the block up front
            head.append(TextBlock(text=memory.text))
        messages = to_messages(records)
        if head:
            messages = with_first_user_blocks(messages, head)
        if hook.text:
            messages = with_last_user_text(messages, hook.text)
        return messages

    # --- the request ---

    def _request(
        self, layers: _Layers, system: str, messages: list[Message], specs: list[ToolSpec]
    ) -> ModelRequest:
        agent, profile = layers.agent, layers.profile
        settings = agent.settings
        thinking = settings.thinking_budget_tokens or THINKING_BUDGET_TOKENS_DEFAULT
        wanted = settings.max_tokens_per_step or layers.budgets.reserve
        hints: list[CacheHint] = []
        if profile.capabilities.prompt_cache == _EXPLICIT_CACHE:
            hints.append(CacheHint(position="system"))
            if specs:
                hints.append(CacheHint(position="tools"))
            if layers.summaries and messages:
                hints.append(CacheHint(position="messages", index=0))
            if messages:
                hints.append(CacheHint(position="messages"))
        return ModelRequest(
            model=agent.model,
            system=system or None,
            messages=messages,
            tools=specs or None,
            max_tokens=min(wanted, profile.max_output_tokens),
            temperature=settings.temperature,
            thinking=ThinkingConfig(budget_tokens=thinking) if settings.thinking else None,
            cache=hints or None,
            metadata=RequestMeta(
                request_id=self._deps.ids.new(ID_PREFIX_REQUEST), agent_id=agent.id
            ),
        )

    async def assemble(
        self, agent_id: AgentId, pending: Sequence[Message] = (), *, retrieve: bool = False
    ) -> Assembly:
        """Build the request and its report.

        Args:
            agent_id: The agent.
            pending: Messages to send that are not in the history yet.
            retrieve: Run the automatic memory search when the run has not answered yet.
        """
        layers = await self._read(agent_id, retrieve=retrieve)
        budgets = layers.budgets
        system_parts = self._system_parts(layers)
        system = SECTION_SEPARATOR.join(part.text for part in system_parts if part.text)
        specs = self._deps.tools.specs_for(agent_id)
        tail = self._tail(layers)
        memory = self._memory_part(layers)
        hook_texts = layers.agent.pending_hook_context
        hook = _Part(
            "hook_context",
            tagged(TAG_HOOK_CONTEXT, SECTION_SEPARATOR.join(hook_texts)) if hook_texts else "",
            len(hook_texts),
            None,
        )
        messages = [*self._messages(layers, tail, memory, hook), *pending]
        tools_tokens = sum(estimate_tokens(spec.model_dump_json()) for spec in specs)
        summary_tokens = blocks_tokens(_summary_blocks(layers.summaries))
        tail_tokens = sum(blocks_tokens(record.content) for record in tail)
        tail_tokens += sum(blocks_tokens(message.content) for message in pending)
        sections = [
            *(part.usage() for part in system_parts),
            SectionUsage(name="tools", tokens=tools_tokens, items=len(specs), budget_tokens=None),
            SectionUsage(
                name="summaries",
                tokens=summary_tokens,
                items=len(layers.summaries),
                budget_tokens=budgets.summaries,
            ),
            SectionUsage(
                name="tail", tokens=tail_tokens, items=len(tail) + len(pending), budget_tokens=None
            ),
            memory.usage(),
            hook.usage(),
        ]
        total = sum(section.tokens for section in sections)
        system_end = sum(part.usage().tokens for part in system_parts)
        tools_end = system_end + tools_tokens
        marks = [system_end, tools_end, tools_end + summary_tokens, total]
        request = self._request(layers, system, messages, specs)
        report = ContextReport(
            model=layers.agent.model,
            window=budgets.window,
            total_tokens=total,
            sections=sections,
            cache_breakpoints=marks if request.cache else [],
            compacted_ranges=[(s.from_seq, s.to_seq) for s in layers.summaries],
            retrieved_memory_ids=list(layers.run.retrieved_memory_ids) if layers.run else [],
        )
        return Assembly(request=request, report=report, budgets=budgets)
