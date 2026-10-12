"""Gateway helpers: cost, backoff, local token estimate."""

import json
import math

from harness.core.types import (
    ContentBlock,
    ImageBlock,
    ModelPrices,
    ModelRequest,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)
from harness.models.consts import (
    BACKOFF_BASE_S,
    BACKOFF_FACTOR,
    BACKOFF_MAX_S,
    ESTIMATE_CHARS_PER_TOKEN,
    ESTIMATE_TOKENS_PER_IMAGE,
    ESTIMATE_TOKENS_PER_MESSAGE,
    TOKENS_PER_MILLION,
)


def cost_usd(usage: Usage, prices: ModelPrices) -> float:
    """Return the cost of one call from its usage and the per-million-token prices."""
    total = (
        usage.input * prices.input
        + usage.output * prices.output
        + usage.cache_read * prices.cache_read
        + usage.cache_write * prices.cache_write
    )
    return total / TOKENS_PER_MILLION


def backoff_s(attempt: int, retry_after_s: float | None = None) -> float:
    """Return the wait before retry number ``attempt`` (0-based); ``retry-after`` wins."""
    if retry_after_s is not None:
        return max(retry_after_s, 0.0)
    return min(BACKOFF_BASE_S * BACKOFF_FACTOR**attempt, BACKOFF_MAX_S)


def _text_tokens(text: str) -> int:
    return math.ceil(len(text) / ESTIMATE_CHARS_PER_TOKEN)


def _block_tokens(block: ContentBlock) -> int:
    if isinstance(block, TextBlock | ThinkingBlock):
        return _text_tokens(block.text)
    if isinstance(block, ToolCallBlock):
        return _text_tokens(block.name) + _text_tokens(json.dumps(block.input))
    if isinstance(block, ToolResultBlock):
        return sum(_block_tokens(inner) for inner in block.content)
    if isinstance(block, ImageBlock):
        return ESTIMATE_TOKENS_PER_IMAGE
    return ESTIMATE_TOKENS_PER_IMAGE


def estimate_tokens(req: ModelRequest) -> int:
    """Estimate the input tokens of a request locally (conservative: it over-counts)."""
    total = 0
    if isinstance(req.system, str):
        total += _text_tokens(req.system)
    elif req.system:
        total += sum(_block_tokens(block) for block in req.system)
    for message in req.messages:
        total += ESTIMATE_TOKENS_PER_MESSAGE
        total += sum(_block_tokens(block) for block in message.content)
    for tool in req.tools or []:
        total += _text_tokens(tool.name) + _text_tokens(tool.description)
        total += _text_tokens(json.dumps(tool.input_schema))
    return total
