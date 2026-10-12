"""The local server: token, listing, sending (phase 2, 02 §5a)."""

from collections.abc import AsyncIterator

import httpx
import pytest

from harness.core.consts import AddressType, AgentStatus, RejectReason
from harness.core.messaging import AgentInfo
from harness.local_server import create_app
from tests.consts import OTHER_AGENT_ID
from tests.fakes.messenger import FakeMessenger

BASE_URL = "http://local.test"
TOKEN = "token-made-for-this-test"
AGENTS_URL = "/v1/agents"
NAME = "reviewer"
MESSAGES_URL = f"/v1/agents/{NAME}/messages"
CONTENT = "the build is red"
CONVERSATION = "conv_from_outside"
HTTP_OK = 200
CLIENT_ERRORS = range(400, 500)
COLLEAGUE = AgentInfo(
    id=OTHER_AGENT_ID,
    name=NAME,
    address=f"agent:{NAME}",
    description="reviews code",
    template=None,
    status=AgentStatus.IDLE,
    accepts_messages=True,
)


@pytest.fixture
def messenger() -> FakeMessenger:
    return FakeMessenger([COLLEAGUE])


@pytest.fixture
async def client(messenger: FakeMessenger) -> AsyncIterator[httpx.AsyncClient]:
    """A client that already carries the right token."""
    transport = httpx.ASGITransport(app=create_app(messenger, TOKEN))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, headers=headers) as made:
        yield made


@pytest.mark.parametrize(
    "authorization", [None, "", "Bearer wrong", TOKEN, f"Basic {TOKEN}", f"Bearer {TOKEN}x"]
)
async def test_nothing_is_done_without_the_token(
    messenger: FakeMessenger, authorization: str | None
) -> None:
    transport = httpx.ASGITransport(app=create_app(messenger, TOKEN))
    headers = {} if authorization is None else {"Authorization": authorization}
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, headers=headers) as anon:
        listed = await anon.get(AGENTS_URL)
        sent = await anon.post(MESSAGES_URL, json={"content": CONTENT})

    assert listed.status_code == sent.status_code
    assert listed.status_code in CLIENT_ERRORS
    assert COLLEAGUE.name not in listed.text
    assert messenger.sent == []
    assert messenger.viewers == []


async def test_the_agents_are_listed(client: httpx.AsyncClient, messenger: FakeMessenger) -> None:
    response = await client.get(AGENTS_URL)

    assert response.status_code == HTTP_OK
    assert response.json() == [COLLEAGUE.model_dump(mode="json")]
    assert [viewer.type for viewer in messenger.viewers] == [AddressType.API]


async def test_a_message_goes_to_the_agent_named_in_the_path(
    client: httpx.AsyncClient, messenger: FakeMessenger
) -> None:
    body = {"content": CONTENT, "expects_reply": True, "conversation_id": CONVERSATION}

    response = await client.post(MESSAGES_URL, json=body)

    ((sender, message),) = messenger.sent
    assert response.status_code == HTTP_OK
    assert response.json()["delivered"] == [NAME]
    assert response.json()["conversation_id"] == CONVERSATION
    assert sender.type is AddressType.API
    assert (message.to, message.content, message.expects_reply) == ([NAME], CONTENT, True)


async def test_the_caller_cannot_choose_who_the_message_is_from(
    client: httpx.AsyncClient, messenger: FakeMessenger
) -> None:
    forged = {"content": CONTENT, "source": {"type": "user", "id": "boss"}}

    response = await client.post(MESSAGES_URL, json=forged)

    assert response.status_code in CLIENT_ERRORS
    assert messenger.sent == []


@pytest.mark.parametrize("reason", list(RejectReason))
async def test_a_message_that_was_not_delivered_is_an_error_with_its_reason(
    client: httpx.AsyncClient, messenger: FakeMessenger, reason: RejectReason
) -> None:
    messenger.refusals[NAME] = reason.value

    response = await client.post(MESSAGES_URL, json={"content": CONTENT})

    assert response.status_code in CLIENT_ERRORS
    assert reason.value in response.json()["detail"]


async def test_unknown_agents_and_busy_limits_get_their_own_status(
    client: httpx.AsyncClient, messenger: FakeMessenger
) -> None:
    statuses: dict[RejectReason, int] = {}
    for reason in (RejectReason.NOT_FOUND, RejectReason.RATE_LIMIT, RejectReason.BLOCKED):
        messenger.refusals[NAME] = reason.value
        statuses[reason] = (await client.post(MESSAGES_URL, json={"content": CONTENT})).status_code

    assert len(set(statuses.values())) == len(statuses)
