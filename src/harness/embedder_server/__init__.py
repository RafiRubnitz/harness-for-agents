"""Embeddings server: BGE-M3 behind a small HTTP API, run as its own process (03 §6.4).

The harness starts it with ``python -m harness.embedder_server`` on 127.0.0.1 and talks
to it only through ``harness.memory``'s ``Embedder``.
"""

from harness.embedder_server.app import create_app
from harness.embedder_server.encoder import BgeEncoder, Encoder

__all__ = ["BgeEncoder", "Encoder", "create_app"]
