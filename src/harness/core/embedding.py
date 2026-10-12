"""Wire format between the harness and its embeddings server (03 §6.4).

Shared by the client in the memory component and by ``harness.embedder_server``, which
runs as a separate process and imports nothing else from the harness.
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

EmbedKind = Literal["document", "query"]

EMBED_PATH: Final = "/embed"
HEALTH_PATH: Final = "/health"
HEALTH_STATUS_OK: Final = "ok"
HEALTH_STATUS_LOADING: Final = "loading"
HEALTH_STATUS_FAILED: Final = "failed"
LOOPBACK_HOST: Final = "127.0.0.1"
EMBEDDER_MODULE: Final = "harness.embedder_server"
EMBEDDER_ARG_PORT: Final = "--port"
EMBEDDER_ARG_MODEL: Final = "--model"


class EmbedRequest(BaseModel):
    """Body of ``POST /embed``."""

    model_config = ConfigDict(extra="forbid")
    texts: list[str]
    kind: EmbedKind


class EmbedResponse(BaseModel):
    """Reply of ``POST /embed``: one vector per text, in order."""

    model_config = ConfigDict(extra="forbid")
    model: str
    vectors: list[list[float]]


class HealthResponse(BaseModel):
    """Reply of ``GET /health``; ``status`` is ``ok`` once the model is loaded."""

    model_config = ConfigDict(extra="forbid")
    status: str
    model: str
    dim: int
