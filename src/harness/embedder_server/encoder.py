"""The model behind the server: a Protocol and the BGE-M3 implementation (03 §6.4)."""

import importlib
from collections.abc import Sequence
from typing import Protocol, cast

from harness.core.embedding import EmbedKind
from harness.embedder_server import consts


class Encoder(Protocol):
    """Turns texts into dense vectors; loading may take long and runs off the event loop."""

    @property
    def model_id(self) -> str:
        """The model this encoder runs."""
        ...

    @property
    def dim(self) -> int:
        """Length of every vector."""
        ...

    def load(self) -> None:
        """Load the model into memory (blocking)."""
        ...

    def encode(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Return one vector per text (blocking)."""
        ...


class _DenseArray(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _FlagModel(Protocol):
    def encode(
        self, sentences: list[str], *, batch_size: int, max_length: int
    ) -> dict[str, _DenseArray]: ...


class BgeEncoder:
    """BGE-M3 through FlagEmbedding, imported only when the model is loaded.

    BGE-M3 needs no different prefix for queries and documents, so ``kind`` is unused.
    """

    def __init__(self, model_id: str = consts.DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._model: _FlagModel | None = None

    @property
    def model_id(self) -> str:
        """The model this encoder runs."""
        return self._model_id

    @property
    def dim(self) -> int:
        """Length of a BGE-M3 dense vector."""
        return consts.BGE_M3_DIM

    def load(self) -> None:
        """Import FlagEmbedding and load the model.

        Raises:
            RuntimeError: The optional ``embedder`` extra is not installed.
        """
        try:
            module = importlib.import_module(consts.FLAG_EMBEDDING_MODULE)
        except ImportError as error:
            raise RuntimeError(consts.MSG_MISSING_EXTRA) from error
        factory = getattr(module, consts.FLAG_MODEL_CLASS)
        self._model = cast("_FlagModel", factory(self._model_id, use_fp16=False))

    def encode(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Return the dense vector of every text.

        Raises:
            RuntimeError: ``load`` was not called.
        """
        if self._model is None:
            raise RuntimeError(consts.MSG_NOT_READY.format(status="not loaded"))
        out = self._model.encode(
            list(texts),
            batch_size=consts.ENCODE_BATCH_SIZE,
            max_length=consts.ENCODE_MAX_LENGTH,
        )
        return out[consts.DENSE_KEY].tolist()
