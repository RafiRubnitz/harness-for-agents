"""Constants of the local server (phase 2, 02 §5a)."""

from typing import Final

APP_TITLE: Final = "harness local server"
LOCALHOST: Final = "127.0.0.1"
LOG_LEVEL: Final = "warning"
TOKEN_SECRET: Final = "local_server_token"
TOKEN_BYTES: Final = 32
# Every caller holds the one local token for now; its id is what ``api:<id>`` names.
TOKEN_ID: Final = "local"
AUTH_SCHEME: Final = "Bearer "
START_POLL_S: Final = 0.01

AGENTS_PATH: Final = "/v1/agents"
MESSAGES_PATH: Final = "/v1/agents/{name_or_id}/messages"

STATUS_UNAUTHORIZED: Final = 401
STATUS_FORBIDDEN: Final = 403
STATUS_NOT_FOUND: Final = 404
STATUS_TOO_MANY: Final = 429

MSG_UNAUTHORIZED: Final = "missing or wrong token"
MSG_REJECTED: Final = "the message was not delivered: {reason}"
MSG_START_FAILED: Final = "the local server could not listen on {host}:{port}"
