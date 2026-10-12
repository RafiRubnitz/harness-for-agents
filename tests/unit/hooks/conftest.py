"""Fixtures for hooks tests: the real hooks manager over a real config and real hook files."""

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.config import FileConfigManager
from harness.core.consts import ConfigLayer, HookEvent, RiskLevel
from harness.core.hook_payloads import BeforeToolCallPayload, HookPayload
from harness.core.hook_types import HookInput, HookRunRecord, HookSpec, MergedOutput
from harness.core.types import AgentId
from harness.core.utils.paths import HarnessHome
from harness.hooks import DefaultHooksManager, HooksDeps
from tests.consts import AGENT_ID, AGENT_NAME, RUN_ID, USER_ID
from tests.fakes.agent_layers import FakeAgentLayerStore
from tests.fakes.clock import FakeClock
from tests.fakes.secrets import FakeSecretStore

TOOL = "shell"
COMMAND = "git status"
CALL_ID = "call_1"
TOOL_REASON = "the user asked for the status"
HOOK_NAME = "guard"
IMPORTS = "from harness.hooks import HookInput, HookOutput\n\n"


@dataclass
class HooksEnv:
    """A hooks manager with its files and everything it reported."""

    manager: DefaultHooksManager
    config: FileConfigManager
    home: HarnessHome
    clock: FakeClock
    work: Path
    load_errors: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    observed: list[HookRunRecord] = field(default_factory=list[HookRunRecord])
    count: int = 0

    def module(self, name: str, body: str) -> Path:
        """Write a Python hook module into the hooks directory."""
        path = self.home.hooks_dir / f"{name}.py"
        path.write_text(IMPORTS + body, encoding="utf-8")
        return path

    def script(self, name: str, body: str) -> Path:
        """Write a hook program (run as an external process) into the home directory."""
        path = self.home.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def hook(
        self,
        run: str,
        event: HookEvent = HookEvent.BEFORE_TOOL_CALL,
        *,
        name: str | None = None,
        **fields: JsonValue,
    ) -> HookSpec:
        """Register a hook; names are made unique."""
        self.count += 1
        spec = HookSpec.model_validate(
            {"name": name or f"{HOOK_NAME}-{self.count}", "event": event, "run": run, **fields}
        )
        self.manager.register(spec)
        return spec

    def answering(self, answer: str, event: HookEvent = HookEvent.BEFORE_TOOL_CALL) -> HookSpec:
        """Register a Python hook that returns ``HookOutput(<answer>)``."""
        self.count += 1
        module = f"answer_{self.count}"
        self.module(
            module,
            f"async def hook(inp: HookInput) -> HookOutput:\n    return HookOutput({answer})\n",
        )
        return self.hook(f"python:{module}:hook", event)

    async def emit(
        self,
        payload: HookPayload | None = None,
        event: HookEvent = HookEvent.BEFORE_TOOL_CALL,
        *,
        agent_id: AgentId | None = AGENT_ID,
        agent_name: str = AGENT_NAME,
    ) -> MergedOutput:
        """Fire an event the way a component does."""
        inp = HookInput(
            event=event,
            agent_id=agent_id,
            agent_name=agent_name,
            run_id=RUN_ID,
            step=1,
            working_dir=str(self.work),
            payload=(payload or tool_call()).model_dump(mode="json"),
            timestamp=self.clock.now(),
        )
        return await self.manager.emit(event, inp)

    async def global_hooks(self, *specs: dict[str, JsonValue]) -> None:
        """Replace the hooks of the global config layer."""
        await self.config.set("hooks.specs", list(specs), ConfigLayer.GLOBAL)


def tool_call(tool: str = TOOL, **tool_input: JsonValue) -> BeforeToolCallPayload:
    """A ``before_tool_call`` payload."""
    return BeforeToolCallPayload(
        tool=tool,
        call_id=CALL_ID,
        input=tool_input or {"command": COMMAND},
        reason=TOOL_REASON,
        read_only=False,
        risk=RiskLevel.MEDIUM,
        run_in_background=False,
    )


@pytest.fixture
def env(tmp_harness_home: HarnessHome, tmp_path: Path, clock: FakeClock) -> Iterator[HooksEnv]:
    """The real hooks manager; hook modules imported by a test are forgotten afterwards."""
    work = tmp_path / "work"
    work.mkdir()
    config = FileConfigManager(
        tmp_harness_home,
        secrets=FakeSecretStore(),
        env={},
        store=FakeAgentLayerStore(),
        username=USER_ID,
    )
    built: list[HooksEnv] = []

    deps = HooksDeps(
        config=config,
        clock=clock,
        home=tmp_harness_home,
        load_failed=lambda spec, message: built[0].load_errors.append((spec.name, message)),
        observe=lambda inp, record: built[0].observed.append(record),
    )
    path_before = list(sys.path)
    built.append(
        HooksEnv(
            manager=DefaultHooksManager(deps),
            config=config,
            home=tmp_harness_home,
            clock=clock,
            work=work,
        )
    )
    yield built[0]
    sys.path[:] = path_before
