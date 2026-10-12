"""Constants of the embeddings server."""

from typing import Final

DEFAULT_MODEL: Final = "BAAI/bge-m3"
DEFAULT_PORT: Final = 8765
BGE_M3_DIM: Final = 1024
ENCODE_BATCH_SIZE: Final = 12
ENCODE_MAX_LENGTH: Final = 8192
DENSE_KEY: Final = "dense_vecs"
FLAG_EMBEDDING_MODULE: Final = "FlagEmbedding"
FLAG_MODEL_CLASS: Final = "BGEM3FlagModel"
APP_TITLE: Final = "harness embeddings"
LOG_LEVEL: Final = "warning"
STATUS_UNAVAILABLE: Final = 503

MSG_NOT_READY: Final = "the embeddings model is not loaded ({status})"
MSG_MISSING_EXTRA: Final = (
    "FlagEmbedding is not installed; install the optional extra: uv sync --extra embedder"
)
PROG: Final = "python -m harness.embedder_server"
HELP_PORT: Final = "port to listen on (127.0.0.1 only)"
HELP_MODEL: Final = "embeddings model id"
