"""The GUI shows who a message came from, and the agent's description (phase 2, 03 §8)."""

from harness.core.consts import AddressType, MessageRole
from harness.core.types import Address, MessageOrigin
from harness.gui.items import Conversation, UserItem, add_user
from tests.consts import OTHER_AGENT_ID, USER_ID
from tests.utils import make_message

TEXT = "the review is done"
SENDER_NAME = "reviewer"


def _shown(origin: MessageOrigin | None) -> UserItem:
    conversation = Conversation()
    message = make_message("msg_1", TEXT, role=MessageRole.USER).model_copy(
        update={"origin": origin}
    )
    add_user(conversation, message)
    (item,) = conversation.items
    assert isinstance(item, UserItem)
    return item


def test_a_message_from_an_agent_carries_the_senders_name() -> None:
    source = Address(type=AddressType.AGENT, id=OTHER_AGENT_ID)

    named = _shown(MessageOrigin(source=source, source_name=SENDER_NAME))
    unnamed = _shown(MessageOrigin(source=source))

    assert named.text == TEXT
    assert named.label is not None
    assert SENDER_NAME in named.label
    assert unnamed.label is not None
    assert OTHER_AGENT_ID in unnamed.label  # falls back to the id


def test_the_users_own_message_has_no_label() -> None:
    from_user = _shown(MessageOrigin(source=Address(type=AddressType.USER, id=USER_ID)))

    assert from_user.label is None
    assert _shown(None).label is None


def test_agents_and_other_sources_are_labelled_differently() -> None:
    agent = _shown(MessageOrigin(source=Address(type=AddressType.AGENT, id=OTHER_AGENT_ID)))
    api = _shown(MessageOrigin(source=Address(type=AddressType.API, id=OTHER_AGENT_ID)))

    assert api.label is not None
    assert agent.label != api.label
