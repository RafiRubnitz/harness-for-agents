"""The file-backed config manager."""

from pathlib import Path

import pytest

from harness.config import (
    FileConfigManager,
    HarnessConfig,
    LimitsConfig,
    PermissionsConfig,
    consts,
    requires_restart,
    template_overrides,
)
from harness.core.consts import ConfigLayer, Decision, ErrorCode, HookEvent, HookScope
from harness.core.errors import ConfigError, SecretMissingError
from harness.core.hook_types import HookSpec
from harness.core.utils.paths import HarnessHome
from tests.consts import AGENT_ID, HEBREW_PLAIN, TEST_API_KEY, USER_ID
from tests.fakes.agent_layers import FakeAgentLayerStore
from tests.fakes.secrets import FakeSecretStore
from tests.unit.config.conftest import SECRET_NAME, ManagerFactory

GLOBAL_RETRIES = 6
PROJECT_K = 4
AGENT_K = 2
ENV_RETRIES = 9
ENCODING = "utf-8"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=ENCODING)


def test_defaults_come_from_the_schema(make_manager: ManagerFactory) -> None:
    manager = make_manager()

    assert manager.config() == HarnessConfig().model_copy(update={"user": manager.config().user})
    assert manager.get("user.id") == USER_ID
    assert manager.get("limits.retries") == consts.MODEL_RETRIES_DEFAULT
    assert manager.resolved().values["limits.retries"].layer is ConfigLayer.DEFAULT


def test_unknown_key_raises(make_manager: ManagerFactory) -> None:
    with pytest.raises(ConfigError):
        make_manager().get("limits.nope")


@pytest.mark.acceptance("06-5.1")
@pytest.mark.acceptance("06-11.3")
@pytest.mark.acceptance("06-11.4")
async def test_layers_override_in_order(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome, project_dir: Path
) -> None:
    _write(
        tmp_harness_home.config_file,
        f"limits:\n  retries: {GLOBAL_RETRIES}\ncontext:\n  auto_retrieve_k: {GLOBAL_RETRIES}\n",
    )
    _write(project_dir / ".harness" / "config.yaml", f"context:\n  auto_retrieve_k: {PROJECT_K}\n")
    manager = make_manager({"HARNESS_LIMITS__RETRIES": str(ENV_RETRIES)})
    await manager.load_agent(AGENT_ID, project_dir)

    assert manager.get("context.auto_retrieve_k") == GLOBAL_RETRIES
    assert manager.get("context.auto_retrieve_k", AGENT_ID) == PROJECT_K
    await manager.set("context.auto_retrieve_k", AGENT_K, ConfigLayer.AGENT, AGENT_ID)
    assert manager.get("context.auto_retrieve_k", AGENT_ID) == AGENT_K
    assert manager.get("limits.retries", AGENT_ID) == ENV_RETRIES
    sources = manager.resolved(AGENT_ID).values
    assert sources["context.auto_retrieve_k"].layer is ConfigLayer.AGENT
    assert sources["limits.retries"].layer is ConfigLayer.ENV


def test_invalid_yaml_reports_file_and_line(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    _write(tmp_harness_home.config_file, "limits:\n  retries: [1, 2\n")

    with pytest.raises(ConfigError) as raised:
        make_manager()

    assert raised.value.file == str(tmp_harness_home.config_file)
    assert raised.value.line is not None


def test_schema_error_points_at_the_offending_line(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    bad_line = 3
    _write(tmp_harness_home.config_file, "gui:\n  theme: x\nlimits:\n  retries: many\n")

    with pytest.raises(ConfigError) as raised:
        make_manager()

    assert raised.value.code is ErrorCode.CONFIG_INVALID
    assert raised.value.line == bad_line + 1


def test_unknown_key_in_file_is_rejected(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    _write(tmp_harness_home.config_file, "limits:\n  retrys: 2\n")

    with pytest.raises(ConfigError):
        make_manager()


@pytest.mark.acceptance("06-6.2")
@pytest.mark.acceptance("06-11.9")
def test_bare_reference_is_a_validation_error(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    key_line = 3
    _write(
        tmp_harness_home.config_file,
        "providers:\n  anthropic:\n    api_key: ${ANTHROPIC_API_KEY}\n",
    )

    with pytest.raises(ConfigError) as raised:
        make_manager()

    assert raised.value.line == key_line


@pytest.mark.acceptance("06-7.2")
async def test_invalid_reload_keeps_previous_config(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    _write(tmp_harness_home.config_file, f"limits:\n  retries: {GLOBAL_RETRIES}\n")
    manager = make_manager()
    rejected: list[ErrorCode] = []

    async def on_error(issues: list[object]) -> None:
        rejected.extend(ErrorCode.CONFIG_INVALID for _ in issues)

    manager.subscribe_errors(on_error)  # type: ignore[arg-type]
    _write(tmp_harness_home.config_file, "limits:\n  retries: many\n")

    await manager.reload()

    assert manager.get("limits.retries") == GLOBAL_RETRIES
    assert rejected


@pytest.mark.acceptance("06-7.3")
async def test_valid_reload_notifies_only_matching_subscribers(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    manager = make_manager()
    limits_changes: list[str] = []
    gui_changes: list[str] = []

    async def on_limits(keys: list[str]) -> None:
        limits_changes.extend(keys)

    async def on_gui(keys: list[str]) -> None:
        gui_changes.extend(keys)

    manager.subscribe(["limits"], on_limits)
    manager.subscribe(["gui"], on_gui)
    _write(tmp_harness_home.config_file, f"limits:\n  retries: {GLOBAL_RETRIES}\n")

    await manager.reload()

    assert limits_changes == ["limits.retries"]
    assert gui_changes == []
    assert manager.section(LimitsConfig).retries == GLOBAL_RETRIES


@pytest.mark.acceptance("06-11.6")
async def test_set_global_writes_file_and_keeps_comments(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    comment = "# keep me"
    _write(tmp_harness_home.config_file, f"{comment}\nlimits:\n  retries: 1\n")
    manager = make_manager()

    await manager.set("limits.retries", GLOBAL_RETRIES, ConfigLayer.GLOBAL)

    assert manager.get("limits.retries") == GLOBAL_RETRIES
    assert comment in tmp_harness_home.config_file.read_text(encoding=ENCODING)
    assert make_manager().get("limits.retries") == GLOBAL_RETRIES


async def test_set_invalid_value_writes_nothing(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    manager = make_manager()

    with pytest.raises(ConfigError):
        await manager.set("limits.retries", "many", ConfigLayer.GLOBAL)

    assert not tmp_harness_home.config_file.exists()
    assert manager.get("limits.retries") == consts.MODEL_RETRIES_DEFAULT


@pytest.mark.parametrize("layer", [ConfigLayer.DEFAULT, ConfigLayer.ENV])
async def test_set_rejects_read_only_layers(
    make_manager: ManagerFactory, layer: ConfigLayer
) -> None:
    with pytest.raises(ConfigError):
        await make_manager().set("context.auto_retrieve_k", AGENT_K, layer)


async def test_agent_layer_goes_through_the_store_and_rejects_global_only_keys(
    make_manager: ManagerFactory, layer_store: FakeAgentLayerStore
) -> None:
    manager = make_manager()
    await manager.load_agent(AGENT_ID, None)

    await manager.set("context.auto_retrieve_k", AGENT_K, ConfigLayer.AGENT, AGENT_ID)
    with pytest.raises(ConfigError):
        await manager.set("limits.retries", GLOBAL_RETRIES, ConfigLayer.AGENT, AGENT_ID)

    assert layer_store.overrides[AGENT_ID] == {"context": {"auto_retrieve_k": AGENT_K}}
    assert manager.get("context.auto_retrieve_k") == consts.AUTO_RETRIEVE_K_DEFAULT


@pytest.mark.acceptance("06-5.1")
@pytest.mark.acceptance("06-11.8")
async def test_agent_allow_rule_does_not_remove_global_deny_rule(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    _write(
        tmp_harness_home.config_file,
        f"permissions:\n  rules:\n    - tool: shell\n      decision: {Decision.DENY.value}\n",
    )
    manager = make_manager()
    await manager.load_agent(AGENT_ID, None)
    allow = [{"tool": "shell", "decision": Decision.ALLOW.value}]

    await manager.set("permissions.rules", allow, ConfigLayer.AGENT, AGENT_ID)

    rules = manager.section(PermissionsConfig, AGENT_ID).rules
    assert [(rule.decision, rule.layer) for rule in rules] == [
        (Decision.DENY, ConfigLayer.GLOBAL),
        (Decision.ALLOW, ConfigLayer.AGENT),
    ]


@pytest.mark.acceptance("06-4.2")
@pytest.mark.acceptance("06-11.10")
async def test_untrusted_project_hooks_are_not_loaded(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome, project_dir: Path
) -> None:
    hook = f"hooks:\n  - name: h\n    event: {HookEvent.BEFORE_TOOL_CALL.value}\n    run: x.py\n"
    _write(project_dir / ".harness" / "hooks.yaml", hook)
    manager = make_manager()

    await manager.load_agent(AGENT_ID, project_dir)

    assert manager.config(AGENT_ID).hooks.specs == []
    assert ErrorCode.PROJECT_UNTRUSTED in [found.code for found in manager.warnings()]
    assert not manager.is_trusted(project_dir)


@pytest.mark.acceptance("06-4.2")
async def test_trusted_project_hooks_are_loaded_with_project_scope(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome, project_dir: Path
) -> None:
    hook = f"hooks:\n  - name: h\n    event: {HookEvent.BEFORE_TOOL_CALL.value}\n    run: x.py\n"
    _write(project_dir / ".harness" / "hooks.yaml", hook)
    manager = make_manager()
    await manager.set("projects.trusted", [str(project_dir)], ConfigLayer.GLOBAL)

    await manager.load_agent(AGENT_ID, project_dir)

    specs = manager.config(AGENT_ID).hooks.specs
    assert [spec.scope for spec in specs] == [HookScope.PROJECT]
    assert manager.config().hooks.specs == []


def test_secret_roundtrip_and_missing(
    make_manager: ManagerFactory, secrets: FakeSecretStore
) -> None:
    manager = make_manager()

    assert manager.secret(SECRET_NAME) == TEST_API_KEY
    assert manager.expand(str(manager.get("providers.anthropic.api_key"))) == TEST_API_KEY
    with pytest.raises(SecretMissingError):
        manager.secret("absent")


async def test_set_secret_never_touches_yaml(
    make_manager: ManagerFactory, secrets: FakeSecretStore, tmp_harness_home: HarnessHome
) -> None:
    manager = make_manager()

    await manager.set_secret("openai", TEST_API_KEY)

    assert secrets.values["openai"] == TEST_API_KEY
    assert not tmp_harness_home.config_file.exists()


def test_validate_reports_missing_secret(tmp_harness_home: HarnessHome) -> None:
    manager = FileConfigManager(
        tmp_harness_home, secrets=FakeSecretStore(), env={}, username=USER_ID
    )

    codes = [found.code for found in manager.validate()]

    assert ErrorCode.CONFIG_SECRET_MISSING in codes


def test_templates_skip_invalid_files(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    _write(tmp_harness_home.templates_dir / "coder.yaml", "name: coder\nsystem_prompt: hi\n")
    _write(tmp_harness_home.templates_dir / "broken.yaml", "name: broken\nbogus: 1\n")
    manager = make_manager()

    names = [template.name for template in manager.templates()]

    assert names == ["coder"]
    assert str(tmp_harness_home.templates_dir / "broken.yaml") in [
        found.file for found in manager.validate()
    ]


async def test_template_is_copied_into_the_agent_layer(
    make_manager: ManagerFactory, tmp_harness_home: HarnessHome
) -> None:
    template_text = (
        "name: coder\nsystem_prompt: hi\n"
        f"hooks:\n  - name: fmt\n    event: {HookEvent.AFTER_TOOL_CALL.value}\n    run: fmt.py\n"
        f"permissions:\n  default: {Decision.ALLOW.value}\n"
        f"config:\n  context:\n    auto_retrieve_k: {AGENT_K}\n"
    )
    _write(tmp_harness_home.templates_dir / "coder.yaml", template_text)
    manager = make_manager()
    await manager.load_agent(AGENT_ID, None)

    await manager.set_agent_layer(AGENT_ID, template_overrides(manager.templates()[0]))

    resolved = manager.config(AGENT_ID)
    assert resolved.context.auto_retrieve_k == AGENT_K
    assert resolved.permissions.default is Decision.ALLOW
    assert [(spec.name, spec.scope) for spec in resolved.hooks.specs] == [("fmt", HookScope.AGENT)]
    assert isinstance(resolved.hooks.specs[0], HookSpec)


async def test_user_profile_roundtrip_in_hebrew(make_manager: ManagerFactory) -> None:
    manager = make_manager()
    assert manager.user_profile() == ""

    await manager.set_user_profile(HEBREW_PLAIN)

    assert manager.user_profile() == HEBREW_PLAIN


def test_requires_restart_follows_the_restart_key_list() -> None:
    restart_key = sorted(consts.RESTART_KEYS)[0]

    assert requires_restart([restart_key])
    assert not requires_restart(["limits.retries"])
