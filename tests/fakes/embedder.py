"""FakeEmbedder: an embeddings server whose notion of meaning the test decides."""

import asyncio
import zlib
from collections.abc import Sequence

from harness.core.embedding import EmbedKind
from harness.core.errors import EmbedderUnavailableError
from harness.memory.consts import EMBED_DIM

FAKE_EMBED_MODEL = "fake-embed-1"
TAUGHT_AXES = 100
MSG_DOWN = "fake embeddings server is down"


class FakeEmbedder:
    """Texts taught as one meaning share a vector; every other text gets its own axis."""

    def __init__(self, model_id: str = FAKE_EMBED_MODEL) -> None:
        self._model_id = model_id
        self.available = True
        self.delay_s = 0.0
        self.calls: list[tuple[list[str], EmbedKind]] = []
        self._meanings: dict[str, list[float]] = {}
        self._next_axis = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def same_meaning(self, *texts: str) -> None:
        """Make these texts embed to the same vector."""
        vector = self._one_hot(self._next_axis)
        self._next_axis += 1
        for text in texts:
            self._meanings[text] = vector

    def close_to(self, text: str, other: str, *, similarity: float) -> None:
        """Make ``text`` embed at the given cosine similarity to ``other``."""
        base = self._vector(other)
        axis = self._next_axis
        self._next_axis += 1
        side = self._one_hot(axis)
        rest = (1.0 - similarity**2) ** 0.5
        self._meanings[text] = [similarity * a + rest * b for a, b in zip(base, side, strict=True)]

    def _one_hot(self, axis: int) -> list[float]:
        vector = [0.0] * EMBED_DIM
        vector[axis] = 1.0
        return vector

    def _vector(self, text: str) -> list[float]:
        if text in self._meanings:
            return self._meanings[text]
        spread = EMBED_DIM - TAUGHT_AXES
        return self._one_hot(TAUGHT_AXES + zlib.crc32(text.encode("utf-8")) % spread)

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Return the taught (or per-text) vectors; raise while ``available`` is off."""
        self.calls.append((list(texts), kind))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if not self.available:
            raise EmbedderUnavailableError(MSG_DOWN)
        return [self._vector(text) for text in texts]

    async def is_available(self) -> bool:
        return self.available
