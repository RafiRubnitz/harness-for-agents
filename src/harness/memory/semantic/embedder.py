"""HTTP client of the embeddings server (03 §6.4)."""

from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from harness.core.embedding import (
    EMBED_PATH,
    HEALTH_PATH,
    HEALTH_STATUS_OK,
    EmbedKind,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)
from harness.core.errors import EmbedderUnavailableError
from harness.memory import consts


class HttpEmbedder:
    """``Embedder`` over the local (or a configured) embeddings server."""

    def __init__(
        self,
        base_url: str,
        *,
        model_id: str,
        dim: int = consts.EMBED_DIM,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model_id = model_id
        self._dim = dim
        self._client = httpx.AsyncClient(
            base_url=base_url, transport=transport, timeout=consts.EMBED_HTTP_TIMEOUT_S
        )

    @property
    def model_id(self) -> str:
        """The configured embeddings model."""
        return self._model_id

    @property
    def dim(self) -> int:
        """Length of every vector."""
        return self._dim

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Return one vector per text; the caller bounds the wait with ``asyncio.timeout``.

        Raises:
            EmbedderUnavailableError: No usable answer came back.
        """
        if not texts:
            return []
        body = EmbedRequest(texts=list(texts), kind=kind).model_dump()
        try:
            response = await self._client.post(EMBED_PATH, json=body)
            response.raise_for_status()
            reply = EmbedResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as error:
            raise EmbedderUnavailableError(consts.MSG_EMBEDDER_DOWN.format(error=error)) from error
        if len(reply.vectors) != len(texts) or any(len(v) != self._dim for v in reply.vectors):
            raise EmbedderUnavailableError(
                consts.MSG_EMBEDDER_BAD_REPLY.format(got=len(reply.vectors), wanted=len(texts))
            )
        return reply.vectors

    async def health(self) -> HealthResponse | None:
        """Return the server's health report, or ``None`` when it does not answer."""
        try:
            response = await self._client.get(HEALTH_PATH, timeout=consts.HEALTH_TIMEOUT_S)
            response.raise_for_status()
            return HealthResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            return None

    async def is_available(self) -> bool:
        """Return whether the server answers with its model loaded."""
        report = await self.health()
        return report is not None and report.status == HEALTH_STATUS_OK

    async def close(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()
