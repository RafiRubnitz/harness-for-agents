"""Entry point of the embeddings server process: ``python -m harness.embedder_server``."""

import argparse

import uvicorn

from harness.core.embedding import EMBEDDER_ARG_MODEL, EMBEDDER_ARG_PORT, LOOPBACK_HOST
from harness.embedder_server import consts
from harness.embedder_server.app import create_app
from harness.embedder_server.encoder import BgeEncoder


def main() -> None:
    """Serve BGE-M3 on the loopback interface until the process is stopped."""
    parser = argparse.ArgumentParser(prog=consts.PROG)
    parser.add_argument(
        EMBEDDER_ARG_PORT, type=int, default=consts.DEFAULT_PORT, help=consts.HELP_PORT
    )
    parser.add_argument(EMBEDDER_ARG_MODEL, default=consts.DEFAULT_MODEL, help=consts.HELP_MODEL)
    args = parser.parse_args()
    port, model = int(args.port), str(args.model)
    uvicorn.run(
        create_app(BgeEncoder(model)), host=LOOPBACK_HOST, port=port, log_level=consts.LOG_LEVEL
    )


if __name__ == "__main__":
    main()
