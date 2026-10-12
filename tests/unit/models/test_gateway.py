"""The gateway: aliases, retry, fallback, cost, call log."""

import asyncio
import json
from pathlib import Path

import pytest

from harness.config import FileConfigManager
from harness.config import consts as config_consts
from harness.core.consts import ConfigLayer, ErrorKind, ProviderId, StopReason, StreamEventType
from harness.core.errors import GatewayError
from harness.core.records import ModelCallRecord
from harness.core.types import ModelRequest, ModelResponse, ToolCallBlock, Usage
from harness.core.utils.paths import HarnessHome
from harness.models import DefaultModelGateway, JsonlCallLog, cost_usd, estimate_tokens
from harness.models import consts as model_consts
from harness.models.reliability import CircuitBreaker
from harness.models.utils import backoff_s
from tests.consts import TEST_API_KEY, USER_ID
from tests.fakes.adapter import FakeAdapter, Outcome, gateway_error, text_response, tool_response
from tests.fakes.clock import FakeClock
from tests.fakes.secrets import FakeSecretStore
from tests.utils import REQUEST_ID, make_request

MAIN = config_consts.MODEL_ALIAS_MAIN
FAST = config_consts.MODEL_ALIAS_FAST
ANSWER = "answer"
USAGE = Usage(input=1000, output=500, cache_read=200, cache_write=100)
RETRY_AFTER_S = 3.0
SHORT_TIMEOUT_MS = 50
TOKEN_COUNT = 77


class RecordingSink:
    """Keeps every call record."""

    def __init__(self) -> None:
        self.records: list[ModelCallRecord] = []
        self.errors: list[GatewayError | None] = []

    async def record_call(
        self,
        record: ModelCallRecord,
        request: ModelRequest,
        response: ModelResponse | None,
        error: GatewayError | None,
    ) -> None:
        self.records.append(record)
        self.errors.append(error)


@pytest.fixture
def config(tmp_harness_home: HarnessHome) -> FileConfigManager:
    return FileConfigManager(
        tmp_harness_home,
        secrets=FakeSecretStore({config_consts.SECRET_ANTHROPIC: TEST_API_KEY}),
        env={},
        username=USER_ID,
    )


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


def _gateway(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink, outcomes: list[Outcome]
) -> tuple[DefaultModelGateway, FakeAdapter]:
    adapter = FakeAdapter(outcomes)
    return DefaultModelGateway(config, {ProviderId.ANTHROPIC: adapter}, clock, [sink]), adapter


def test_resolve_alias_id_and_unknown(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    gateway, _ = _gateway(config, clock, sink, [])

    by_alias = gateway.resolve(MAIN)

    assert by_alias.id == config_consts.MODEL_ID_MAIN
    assert by_alias.fallback == [FAST]
    assert gateway.resolve(config_consts.MODEL_ID_MAIN) == by_alias
    assert {profile.alias for profile in gateway.list_models()} == set(config.config().models)
    with pytest.raises(GatewayError) as raised:
        gateway.resolve("no-such-model")
    assert raised.value.kind is ErrorKind.INVALID_REQUEST


@pytest.mark.acceptance("01-10.5")
async def test_complete_fills_cost_model_and_records_the_call(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    gateway, adapter = _gateway(config, clock, sink, [text_response(ANSWER, USAGE)])
    profile = gateway.resolve(MAIN)

    response = await gateway.complete(make_request(model=MAIN))

    assert response.cost_usd == cost_usd(USAGE, profile.prices)
    assert response.cost_usd > 0
    assert response.model_used == profile.id
    assert adapter.requests[0].model == profile.id
    assert adapter.requests[0].max_tokens == profile.max_output_tokens
    record = sink.records[0]
    assert record.id == REQUEST_ID
    assert (record.model_alias, record.model_used) == (MAIN, profile.id)
    assert (record.usage, record.cost_usd) == (USAGE, response.cost_usd)
    assert record.stop_reason is StopReason.END


@pytest.mark.acceptance("01-10.4")
async def test_rate_limit_is_retried_after_retry_after(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    outcomes: list[Outcome] = [
        gateway_error(ErrorKind.RATE_LIMIT, RETRY_AFTER_S),
        gateway_error(ErrorKind.OVERLOADED),
        text_response(ANSWER),
    ]
    gateway, adapter = _gateway(config, clock, sink, outcomes)

    response = await gateway.complete(make_request(model=MAIN))

    assert response.model_used == config_consts.MODEL_ID_MAIN
    assert clock.sleeps == [RETRY_AFTER_S, backoff_s(1)]
    assert len(adapter.requests) == len(outcomes)
    assert len(sink.records) == 1


@pytest.mark.acceptance("01-10.4")
async def test_persistent_failure_falls_back_to_the_next_model(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    attempts = config_consts.MODEL_RETRIES_DEFAULT + 1
    outcomes: list[Outcome] = [gateway_error(ErrorKind.OVERLOADED) for _ in range(attempts)]
    outcomes.append(text_response(ANSWER))
    gateway, adapter = _gateway(config, clock, sink, outcomes)
    gateway._breakers[ProviderId.ANTHROPIC] = _never_open()

    response = await gateway.complete(make_request(model=MAIN))

    assert response.model_used == config_consts.MODEL_ID_FAST
    assert [request.model for request in adapter.requests] == [
        *[config_consts.MODEL_ID_MAIN] * attempts,
        config_consts.MODEL_ID_FAST,
    ]
    assert sink.records[0].model_alias == MAIN
    assert sink.records[0].model_used == config_consts.MODEL_ID_FAST


@pytest.mark.parametrize(
    ("alias", "provider"),
    [
        (config_consts.MODEL_ALIAS_MAIN, ProviderId.ANTHROPIC),
        (config_consts.MODEL_ALIAS_GPT, ProviderId.OPENAI),
        (config_consts.MODEL_ALIAS_GPT_MINI, ProviderId.OPENAI),
        (config_consts.MODEL_ALIAS_DEEPSEEK, ProviderId.DEEPSEEK),
        (config_consts.MODEL_ALIAS_DEEPSEEK_REASONER, ProviderId.DEEPSEEK),
    ],
)
def test_every_provider_has_a_model_out_of_the_box(
    config: FileConfigManager,
    clock: FakeClock,
    sink: RecordingSink,
    alias: str,
    provider: ProviderId,
) -> None:
    gateway, _ = _gateway(config, clock, sink, [])

    assert gateway.resolve(alias).provider is provider


def _never_open() -> CircuitBreaker:
    return CircuitBreaker(threshold=10**6)


@pytest.mark.parametrize(
    "kind", [ErrorKind.AUTH, ErrorKind.INVALID_REQUEST, ErrorKind.CONTEXT_OVERFLOW]
)
async def test_non_retryable_errors_fail_immediately_without_fallback(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink, kind: ErrorKind
) -> None:
    gateway, adapter = _gateway(config, clock, sink, [gateway_error(kind)])

    with pytest.raises(GatewayError) as raised:
        await gateway.complete(make_request(model=MAIN))

    assert raised.value.kind is kind
    assert len(adapter.requests) == 1
    assert clock.sleeps == []
    assert sink.errors == [raised.value]
    assert sink.records[0].cost_usd == 0


async def test_failure_after_partial_output_is_not_retried(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    gateway, adapter = _gateway(config, clock, sink, [text_response(ANSWER)])
    adapter.fail_after = (2, gateway_error(ErrorKind.OVERLOADED))
    seen: list[StreamEventType] = []

    async def consume() -> None:
        async for event in gateway.stream(make_request(model=MAIN)):
            seen.append(event.type)

    with pytest.raises(GatewayError):
        await consume()

    assert StreamEventType.TEXT_DELTA in seen
    assert len(adapter.requests) == 1


async def test_circuit_breaker_blocks_a_failing_provider(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    threshold = model_consts.BREAKER_FAILURE_THRESHOLD
    outcomes: list[Outcome] = [gateway_error(ErrorKind.OVERLOADED) for _ in range(threshold)]
    outcomes.append(text_response(ANSWER))
    gateway, adapter = _gateway(config, clock, sink, outcomes)
    await config.set("limits.retries", threshold, ConfigLayer.GLOBAL)

    with pytest.raises(GatewayError):
        await gateway.complete(make_request(model=MAIN))
    blocked_calls = len(adapter.requests)
    with pytest.raises(GatewayError):
        await gateway.complete(make_request(model=MAIN))

    assert blocked_calls == threshold
    assert len(adapter.requests) == threshold
    clock.advance(model_consts.BREAKER_COOLDOWN_S)
    assert (await gateway.complete(make_request(model=MAIN))).model_used


async def test_idle_stream_times_out(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    gateway, adapter = _gateway(config, clock, sink, [])
    adapter.hang = True
    await config.set("limits.timeout_ms", SHORT_TIMEOUT_MS, ConfigLayer.GLOBAL)
    await config.set("limits.retries", 0, ConfigLayer.GLOBAL)
    await config.set("models.main.fallback", [], ConfigLayer.GLOBAL)

    with pytest.raises(GatewayError) as raised:
        await gateway.complete(make_request(model=MAIN))

    assert raised.value.kind is ErrorKind.TIMEOUT


@pytest.mark.acceptance("01-10.6")
async def test_cancelling_the_task_stops_the_stream(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    gateway, adapter = _gateway(config, clock, sink, [])
    adapter.hang = True
    task = asyncio.create_task(gateway.complete(make_request(model=MAIN)))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)


async def test_tool_call_passes_through_unchanged(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    call = ToolCallBlock(id="call_1", name="mcp.github.create_issue", input={"title": "x"})
    gateway, _ = _gateway(config, clock, sink, [tool_response(call)])

    response = await gateway.complete(make_request(model=MAIN))

    assert response.content == [call]
    assert response.stop_reason is StopReason.TOOL_CALL


async def test_count_tokens_prefers_the_provider_and_falls_back_to_an_estimate(
    config: FileConfigManager, clock: FakeClock, sink: RecordingSink
) -> None:
    gateway, adapter = _gateway(config, clock, sink, [])
    request = make_request(model=MAIN)

    estimated = await gateway.count_tokens(request)
    adapter.token_count = TOKEN_COUNT
    counted = await gateway.count_tokens(request)

    assert estimated == estimate_tokens(request)
    assert estimated > 0
    assert counted == TOKEN_COUNT


@pytest.mark.acceptance("01-10.5")
async def test_jsonl_call_log_holds_request_response_and_usage(
    config: FileConfigManager, clock: FakeClock, tmp_path: Path
) -> None:
    path = tmp_path / "logs" / model_consts.CALL_LOG_FILENAME
    adapter = FakeAdapter([text_response(ANSWER, USAGE)])
    gateway = DefaultModelGateway(
        config, {ProviderId.ANTHROPIC: adapter}, clock, [JsonlCallLog(path)]
    )

    response = await gateway.complete(make_request(model=MAIN))

    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert len(lines) == 1
    assert entry["call"]["id"] == REQUEST_ID
    assert entry["call"]["usage"] == USAGE.model_dump()
    assert entry["call"]["cost_usd"] == response.cost_usd
    assert entry["request"]["model"] == MAIN
    assert entry["response"]["content"][0]["text"] == ANSWER
    assert entry["error"] is None
