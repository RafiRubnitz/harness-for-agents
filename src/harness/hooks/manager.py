"""DefaultHooksManager: find the hooks of an event, run them, merge the answers (05)."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from harness.config import ConfigManager, HooksConfig
from harness.core.consts import ErrorCode, HookEvent, HookOnError
from harness.core.errors import AgentNotFoundError, HookError
from harness.core.hook_types import HookInput, HookOutput, HookRunRecord, HookSpec, MergedOutput
from harness.core.records import AgentState
from harness.core.types import AgentId, Clock
from harness.core.utils.paths import HarnessHome
from harness.hooks import consts
from harness.hooks.merge import Merge
from harness.hooks.runners import ModuleCache, Runner, build
from harness.hooks.utils import matches, template_in

_HOOKS_KEY = "hooks"


@dataclass(frozen=True, slots=True)
class HooksDeps:
    """What the hooks manager is built from.

    Attributes:
        agent: Looks an agent up (for ``match: {template: ..}``); optional.
        load_failed: Told once about every hook that could not be loaded.
        observe: Told about every hook run, for callers that do not show ``fired`` themselves.
    """

    config: ConfigManager
    clock: Clock
    home: HarnessHome
    agent: Callable[[AgentId], AgentState] | None = None
    load_failed: Callable[[HookSpec, str], None] | None = None
    observe: Callable[[HookInput, HookRunRecord], None] | None = None


@dataclass(frozen=True, slots=True)
class _Failure:
    code: ErrorCode
    message: str


class DefaultHooksManager:
    """Runs the hooks of the global, project and agent scopes, in that order."""

    def __init__(self, deps: HooksDeps) -> None:
        self._deps = deps
        self._registered: list[HookSpec] = []
        self._modules = ModuleCache(deps.home.hooks_dir)
        self._reported: dict[str, str] = {}
        self._background: set[asyncio.Task[None]] = set()
        deps.config.subscribe([_HOOKS_KEY], self._on_config_change)

    async def _on_config_change(self, keys: list[str]) -> None:
        await self.reload()

    # --- registry --------------------------------------------------------------------

    def register(self, hook: HookSpec) -> None:
        """Add a hook at runtime; it runs before the configured ones.

        Raises:
            HookError: A hook of that name is already registered.
        """
        if any(spec.name == hook.name for spec in self._registered):
            message = consts.MSG_DUPLICATE_HOOK.format(name=hook.name)
            raise HookError(message, hook=hook.name, code=ErrorCode.HOOK_LOAD_FAILED)
        self._registered.append(hook)

    def unregister(self, name: str) -> None:
        """Remove a hook added with ``register``; unknown names are ignored."""
        self._registered = [spec for spec in self._registered if spec.name != name]

    async def reload(self) -> None:
        """Forget imported hook modules and check that every global hook still loads."""
        self._modules.clear()
        self._reported.clear()
        for spec in self.list():
            self._runner(spec)

    async def close(self) -> None:
        """Stop the async hooks that are still running."""
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- loading ---------------------------------------------------------------------

    def _load_failed(self, spec: HookSpec, message: str) -> None:
        """Report a hook that cannot be loaded, once per distinct problem."""
        if self._reported.get(spec.name) == message:
            return
        self._reported[spec.name] = message
        logger.warning("hook '{}' was not loaded: {}", spec.name, message)
        if self._deps.load_failed is not None:
            self._deps.load_failed(spec, message)

    def _runner(self, spec: HookSpec) -> Runner | None:
        """Build and load the hook's runner; a hook that cannot be loaded is skipped."""
        settings = self._deps.config.section(HooksConfig)
        try:
            runner = build(spec, settings, self._modules, self._deps.home.root)
            runner.check()
        except HookError as error:
            self._load_failed(spec, error.message)
            return None
        self._reported.pop(spec.name, None)
        return runner

    def _template(self, inp: HookInput) -> str | None:
        if inp.agent_id is not None and self._deps.agent is not None:
            try:
                return self._deps.agent(inp.agent_id).template
            except AgentNotFoundError:
                pass
        return template_in(inp.payload)

    # --- running ---------------------------------------------------------------------

    def _timeout_s(self, spec: HookSpec) -> float:
        settings = self._deps.config.section(HooksConfig)
        default = settings.async_timeout_ms if spec.is_async else settings.timeout_ms
        return (spec.timeout_ms or default) / consts.MS_PER_S

    async def _answer(
        self, spec: HookSpec, runner: Runner, inp: HookInput
    ) -> HookOutput | _Failure:
        """Run one hook under its timeout; whatever goes wrong becomes a ``_Failure``."""
        timeout_s = self._timeout_s(spec)
        try:
            async with asyncio.timeout(timeout_s):
                return await runner.run(inp)
        except TimeoutError:
            ms = round(timeout_s * consts.MS_PER_S)
            return _Failure(
                ErrorCode.HOOK_TIMEOUT, consts.MSG_TIMEOUT.format(hook=spec.name, ms=ms)
            )
        except HookError as error:
            return _Failure(error.code, error.message)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # a hook must never bring the agent down
            message = consts.MSG_RAISED.format(hook=spec.name, error=repr(error))
            return _Failure(ErrorCode.HOOK_FAILED, message)

    def _on_error(self, spec: HookSpec, failure: _Failure) -> HookOutput:
        """What a failed hook counts as (05 §7): ``on_error``, strict for safety events."""
        strict = spec.event in consts.DENY_ON_ERROR_EVENTS
        policy = spec.on_error or (HookOnError.DENY if strict else HookOnError.IGNORE)
        logger.warning("hook failed ({}, on_error={}): {}", failure.code, policy, failure.message)
        decision = consts.DECISION_ON_ERROR[policy]
        return HookOutput(decision=decision, reason=failure.message if decision else None)

    async def _run(self, spec: HookSpec, runner: Runner, inp: HookInput, merge: Merge) -> None:
        """Run one blocking hook and fold its answer into ``merge``."""
        started = self._deps.clock.now()
        answer = await self._answer(spec, runner, inp)
        if isinstance(answer, _Failure) and answer.code is ErrorCode.HOOK_LOAD_FAILED:
            self._load_failed(spec, answer.message)
            return
        if isinstance(answer, HookOutput) and answer.updated_payload is not None:
            problem = merge.change_payload(spec.name, answer.updated_payload)
            if problem is not None:
                answer = _Failure(ErrorCode.HOOK_INVALID_OUTPUT, problem)
        failure = answer if isinstance(answer, _Failure) else None
        output = self._on_error(spec, failure) if failure else answer
        assert isinstance(output, HookOutput)  # noqa: S101 - narrowed for the type checker
        record = self._record(spec, inp, output, started, failure)
        merge.add(record, output)
        if self._deps.observe is not None:
            self._deps.observe(inp, record)

    def _record(
        self,
        spec: HookSpec,
        inp: HookInput,
        output: HookOutput,
        started: datetime,
        failure: _Failure | None,
    ) -> HookRunRecord:
        elapsed = self._deps.clock.now() - started
        return HookRunRecord(
            hook=spec.name,
            event=spec.event,
            agent_id=inp.agent_id,
            decision=output.decision,
            reason=output.reason,
            duration_ms=round(elapsed.total_seconds() * consts.MS_PER_S),
            error=failure.code if failure else None,
            suppressed=output.suppress_output,
        )

    def _spawn(self, spec: HookSpec, runner: Runner, inp: HookInput) -> None:
        """Start an async hook: nobody waits for it and its decision is ignored."""

        async def run() -> None:
            answer = await self._answer(spec, runner, inp)
            if isinstance(answer, _Failure):
                logger.warning("async hook failed ({}): {}", answer.code, answer.message)

        task = asyncio.create_task(run(), name=f"hook:{spec.name}")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def emit(self, event: HookEvent, inp: HookInput) -> MergedOutput:
        """Run every hook that matches ``event`` and return the merged output.

        A ``deny`` ends the chain; each hook sees the payload as the previous ones left it.
        """
        merge = Merge(event, inp.payload)
        template = self._template(inp)
        for spec in self.list(event, inp.agent_id):
            runner = self._runner(spec)  # first, so that a broken hook is always reported
            if runner is None or not matches(spec, inp, template):
                continue
            current = inp.model_copy(update={"payload": merge.payload})
            if spec.is_async:
                self._spawn(spec, runner, current)
                continue
            await self._run(spec, runner, current, merge)
            if merge.denied:
                break
        return merge.result()

    def list(
        self, event: HookEvent | None = None, agent_id: AgentId | None = None
    ) -> list[HookSpec]:
        """The hooks that apply to an agent (or globally), in the order they run."""
        configured = self._deps.config.section(HooksConfig, agent_id).specs
        return [spec for spec in (*self._registered, *configured) if event in (None, spec.event)]
