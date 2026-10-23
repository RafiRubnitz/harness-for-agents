"""The FastAPI app of the local server: list agents, send a message to one.

Only ``harness.core`` is used: a message goes through the ``Messenger`` (and from there
through ``AgentManager.send``), so it is always stored in the database before the agent
is woken, and its origin is set here from the token - never from the request body.
"""

import secrets
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from harness.core.consts import AddressType, RejectReason
from harness.core.messaging import AgentInfo, Messenger, SendMessageInput, SendMessageOutput
from harness.core.types import Address
from harness.local_server import consts

_STATUS_OF = {
    RejectReason.NOT_FOUND.value: consts.STATUS_NOT_FOUND,
    RejectReason.ARCHIVED.value: consts.STATUS_NOT_FOUND,
    RejectReason.RATE_LIMIT.value: consts.STATUS_TOO_MANY,
}


class MessageBody(BaseModel):
    """Body of ``POST /v1/agents/{name_or_id}/messages``."""

    model_config = ConfigDict(extra="forbid")
    content: str
    expects_reply: bool = False
    conversation_id: str | None = None


def create_app(messenger: Messenger, token: str) -> FastAPI:
    """Build the server; every request must carry ``Authorization: Bearer <token>``."""
    app = FastAPI(title=consts.APP_TITLE)
    caller = Address(type=AddressType.API, id=consts.TOKEN_ID)

    def check(authorization: str | None) -> None:
        scheme, given = (authorization or "")[: len(consts.AUTH_SCHEME)], ""
        if scheme == consts.AUTH_SCHEME:
            given = (authorization or "")[len(consts.AUTH_SCHEME) :]
        if not given or not secrets.compare_digest(given.encode(), token.encode()):
            raise HTTPException(
                status_code=consts.STATUS_UNAUTHORIZED, detail=consts.MSG_UNAUTHORIZED
            )

    @app.get(consts.AGENTS_PATH)
    async def agents(  # pyright: ignore[reportUnusedFunction]
        authorization: Annotated[str | None, Header()] = None,
    ) -> list[AgentInfo]:
        check(authorization)
        return messenger.list_agents(caller)

    @app.post(consts.MESSAGES_PATH)
    async def send(  # pyright: ignore[reportUnusedFunction]
        name_or_id: str,
        body: MessageBody,
        authorization: Annotated[str | None, Header()] = None,
    ) -> SendMessageOutput:
        check(authorization)
        message = SendMessageInput(
            to=[name_or_id],
            content=body.content,
            expects_reply=body.expects_reply,
            conversation_id=body.conversation_id,
        )
        output = await messenger.send(caller, message)
        if not output.delivered:
            reason = output.rejected[0][1]
            raise HTTPException(
                status_code=_STATUS_OF.get(reason, consts.STATUS_FORBIDDEN),
                detail=consts.MSG_REJECTED.format(reason=reason),
            )
        return output

    return app
