"""Compaction: the oldest messages become a summary; the originals stay in history (07 §5)."""

from collections.abc import Sequence

from loguru import logger

from harness.config import ContextConfig
from harness.context import history
from harness.context.consts import (
    LINE_SEPARATOR,
    MSG_COMPACTION_DENIED,
    MSG_COMPACTION_FAILED,
    MSG_EMPTY_SUMMARY,
    MSG_NOTHING_TO_COMPACT,
    REASON_JOINER,
    SECTION_SEPARATOR,
    SUMMARY_BLOCK,
    SUMMARY_CHUNK_WINDOW_SHARE,
    SUMMARY_INSTRUCTIONS_HEADER,
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_PREVIOUS_HEADER,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_TRANSCRIPT_HEADER,
    TAG_SUMMARY,
)
from harness.context.types import Budgets, ContextDeps, budgets_for
from harness.context.utils import estimate_tokens, plain_text
from harness.core.consts import (
    ID_PREFIX_REQUEST,
    ID_PREFIX_SUMMARY,
    AgentEventType,
    CompactReason,
    Decision,
    HookEvent,
    RequestPurpose,
)
from harness.core.errors import ContextError, GatewayError
from harness.core.events import ContextCompacted
from harness.core.hook_payloads import ContextCompactPayload
from harness.core.hook_types import HookInput
from harness.core.records import AgentEventRecord, AgentRecord, MessageRecord, SummaryRecord
from harness.core.types import AgentId, Message, ModelRequest, RequestMeta, TextBlock

_INSTRUCTIONS_KEY = "instructions"


class Compactor:
    """Picks the range, asks the hooks, summarises, and stores the summary."""

    def __init__(self, deps: ContextDeps) -> None:
        self._deps = deps

    # --- the range ---

    async def _unsummarised(self, agent_id: AgentId) -> list[MessageRecord]:
        summaries = await self._deps.memory.get_summaries(agent_id)
        covered = max((s.to_seq for s in summaries), default=-1)
        return await self._deps.memory.get_messages(agent_id, from_seq=covered + 1)

    @staticmethod
    def _head(
        records: Sequence[MessageRecord], budgets: Budgets, limit: int, reason: CompactReason
    ) -> list[MessageRecord]:
        """The oldest records, up to a safe cut that leaves the tail in the window."""
        sized = history.shrink_results(records, limit)
        target = history.tail_start(sized, budgets.tail_keep)
        if target == 0 and reason is CompactReason.MANUAL:
            target = len(records)  # the user asked: compact whatever can be compacted
        elif target == 0 and reason is CompactReason.OVERFLOW:
            # the provider refused a request the estimate thought was small: trust the
            # provider and keep only the newest message
            target = len(records) - 1
        cut = history.safe_cut(records, target) if target > 0 else None
        if cut is None:
            raise ContextError(MSG_NOTHING_TO_COMPACT)
        return list(records[:cut])

    # --- hooks ---

    async def _ask_hooks(
        self, agent: AgentRecord, head: Sequence[MessageRecord], reason: CompactReason
    ) -> str | None:
        """Fire ``context_compact``; return the extra summary instructions, if any.

        Raises:
            ContextError: A hook denied the compaction.
        """
        run = await self._deps.memory.open_run(agent.id)
        payload = ContextCompactPayload(from_seq=head[0].seq, to_seq=head[-1].seq, reason=reason)
        out = await self._deps.hooks.emit(
            HookEvent.CONTEXT_COMPACT,
            HookInput(
                event=HookEvent.CONTEXT_COMPACT,
                agent_id=agent.id,
                agent_name=agent.name,
                run_id=run.id if run else None,
                step=run.step_count if run else None,
                working_dir=agent.settings.working_dir,
                payload=payload.model_dump(mode="json"),
                timestamp=self._deps.clock.now(),
            ),
        )
        if out.decision is Decision.DENY:
            raise ContextError(
                MSG_COMPACTION_DENIED.format(reasons=REASON_JOINER.join(out.reasons))
            )
        instructions = out.payload.get(_INSTRUCTIONS_KEY)
        return instructions if isinstance(instructions, str) and instructions.strip() else None

    # --- the summary model ---

    async def _ask(self, agent_id: AgentId, model: str, prompt: str) -> str:
        profile = self._deps.gateway.resolve(model)
        response = await self._deps.gateway.complete(
            ModelRequest(
                model=model,
                system=SUMMARY_SYSTEM_PROMPT,
                messages=[Message(role="user", content=[TextBlock(text=prompt)])],
                max_tokens=min(SUMMARY_MAX_OUTPUT_TOKENS, profile.max_output_tokens),
                metadata=RequestMeta(
                    request_id=self._deps.ids.new(ID_PREFIX_REQUEST),
                    agent_id=agent_id,
                    purpose=RequestPurpose.COMPACTION,
                ),
            )
        )
        text = plain_text(response.content).strip()
        if not text:
            raise ContextError(MSG_EMPTY_SUMMARY)
        return text

    async def _fold(
        self, agent_id: AgentId, model: str, lines: Sequence[str], instructions: str | None
    ) -> str:
        """Summarise ``lines`` with one model, chunk by chunk when they are too long."""
        window = self._deps.gateway.resolve(model).context_window
        summary = ""
        for chunk in history.chunked(lines, int(window * SUMMARY_CHUNK_WINDOW_SHARE)):
            parts = [f"{SUMMARY_TRANSCRIPT_HEADER}\n{LINE_SEPARATOR.join(chunk)}"]
            if summary:
                parts.insert(0, f"{SUMMARY_PREVIOUS_HEADER}\n{summary}")
            if instructions:
                parts.append(f"{SUMMARY_INSTRUCTIONS_HEADER}\n{instructions}")
            summary = await self._ask(agent_id, model, SECTION_SEPARATOR.join(parts))
        return summary

    async def _summarise(
        self,
        agent: AgentRecord,
        settings: ContextConfig,
        lines: Sequence[str],
        instructions: str | None,
    ) -> tuple[str, str]:
        """Return the summary and the model that wrote it.

        The configured summary model is tried first and the agent's own model second, so
        a missing ``cheap`` alias does not stop a compaction.

        Raises:
            ContextError: No model produced a summary.
        """
        failure: Exception | None = None
        for model in dict.fromkeys([settings.summarizer_model, agent.model]):
            try:
                return await self._fold(agent.id, model, lines, instructions), model
            except (GatewayError, ContextError) as error:
                logger.warning("summary model {} failed: {}", model, error.message)
                failure = error
        raise ContextError(MSG_COMPACTION_FAILED.format(error=failure)) from failure

    # --- storing ---

    async def _merge(
        self, agent: AgentRecord, settings: ContextConfig, budgets: Budgets
    ) -> SummaryRecord | None:
        """Fold every active summary into one when together they are over their budget."""
        active = sorted(await self._deps.memory.get_summaries(agent.id), key=lambda s: s.from_seq)
        if len(active) <= 1 or sum(estimate_tokens(s.text) for s in active) <= budgets.summaries:
            return None
        lines = [
            SUMMARY_BLOCK.format(tag=TAG_SUMMARY, from_seq=s.from_seq, to_seq=s.to_seq, body=s.text)
            for s in active
        ]
        text, model = await self._summarise(agent, settings, lines, None)
        merged = SummaryRecord(
            id=self._deps.ids.new(ID_PREFIX_SUMMARY),
            agent_id=agent.id,
            from_seq=active[0].from_seq,
            to_seq=active[-1].to_seq,
            text=text,
            level=max(s.level for s in active) + 1,
            reason=CompactReason.THRESHOLD,
            model_used=model,
            token_count=estimate_tokens(text),
            created_at=self._deps.clock.now(),
        )
        await self._deps.memory.save_summary(merged)
        for old in active:
            await self._deps.memory.save_summary(
                old.model_copy(update={"superseded_by": merged.id})
            )
        return merged

    async def _announce(self, agent_id: AgentId, summary: SummaryRecord, shown_id: str) -> None:
        shown = ContextCompacted(
            from_seq=summary.from_seq, to_seq=summary.to_seq, summary_id=shown_id
        )
        run = await self._deps.memory.open_run(agent_id)
        await self._deps.memory.append_event(
            AgentEventRecord(
                agent_id=agent_id,
                run_id=run.id if run else None,
                after_seq=await self._deps.memory.last_seq(agent_id),
                type=AgentEventType.COMPACTED,
                payload=shown.model_dump(mode="json"),
                created_at=self._deps.clock.now(),
            )
        )
        if self._deps.notify is not None:
            self._deps.notify.compacted(agent_id, summary.model_copy(update={"id": shown_id}))

    async def compact(self, agent_id: AgentId, reason: CompactReason) -> SummaryRecord:
        """Summarise the oldest unsummarised messages.

        Raises:
            ContextError: Nothing can be compacted, a hook denied it, or no model answered.
        """
        deps = self._deps
        agent = await deps.memory.get_agent(agent_id)
        settings = deps.config.section(ContextConfig, agent_id)
        budgets = budgets_for(deps.gateway.resolve(agent.model), settings)
        records = await self._unsummarised(agent_id)
        head = self._head(records, budgets, settings.tool_results.truncate_tokens, reason)
        instructions = await self._ask_hooks(agent, head, reason)
        text, model = await self._summarise(agent, settings, history.transcript(head), instructions)
        summary = SummaryRecord(
            id=deps.ids.new(ID_PREFIX_SUMMARY),
            agent_id=agent_id,
            from_seq=head[0].seq,
            to_seq=head[-1].seq,
            text=text,
            reason=reason,
            model_used=model,
            token_count=estimate_tokens(text),
            created_at=deps.clock.now(),
        )
        await deps.memory.save_summary(summary)
        try:
            merged = await self._merge(agent, settings, budgets)
        except ContextError as error:
            logger.warning("summaries were not merged: {}", error.message)
            merged = None
        await self._announce(agent_id, summary, (merged or summary).id)
        return merged or summary
