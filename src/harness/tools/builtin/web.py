"""The ``web`` group: ``web_scrape`` and ``web_search`` through Firecrawl (08 §4.4).

Without a Firecrawl key ``web_scrape`` falls back to a plain fetch converted to Markdown
and ``web_search`` is not available.
"""

from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue

from harness.config import ConfigManager
from harness.core.consts import Decision, ErrorCode, RiskLevel, ToolSource
from harness.core.errors import SecretMissingError
from harness.core.records import ToolOutput
from harness.tools import consts
from harness.tools.builtin.markdown import html_to_markdown
from harness.tools.function_tool import FunctionTool, error_output, text_output, tool
from harness.tools.types import ToolTraits


class WebClient:
    """Talks to Firecrawl, or straight to the page when there is no key."""

    def __init__(
        self, config: ConfigManager, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._transport = transport

    def _key(self) -> str | None:
        if self._config.config().web.provider != consts.WEB_PROVIDER_FIRECRAWL:
            return None
        try:
            return self._config.secret(consts.WEB_SECRET_NAME)
        except SecretMissingError:
            return None

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=consts.WEB_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": consts.WEB_USER_AGENT},
        )

    async def _firecrawl(self, path: str, key: str, body: dict[str, JsonValue]) -> JsonValue:
        async with self._client() as client:
            response = await client.post(
                consts.FIRECRAWL_BASE_URL + path,
                json=body,
                headers={"Authorization": f"Bearer {key}"},
            )
        response.raise_for_status()
        answer: JsonValue = response.json()
        if not isinstance(answer, dict) or not answer.get("success", True):
            raise httpx.HTTPError(consts.MSG_WEB_BAD_ANSWER)
        return answer.get("data")

    async def scrape(self, url: str, formats: list[str]) -> ToolOutput:
        """Return the page as Markdown."""
        key = self._key()
        if key is None:
            return await self._fetch(url)
        data = await self._firecrawl(
            consts.FIRECRAWL_SCRAPE, key, {"url": url, "formats": [*formats]}
        )
        if not isinstance(data, dict):
            raise httpx.HTTPError(consts.MSG_WEB_BAD_ANSWER)
        parts = [str(data[name]) for name in formats if data.get(name)]
        return text_output(
            "\n\n".join(parts),
            metadata={consts.META_URL: url, consts.META_VIA: consts.WEB_PROVIDER_FIRECRAWL},
        )

    async def _fetch(self, url: str) -> ToolOutput:
        async with self._client() as client:
            response = await client.get(url)
        response.raise_for_status()
        kind = response.headers.get("content-type", "")
        body = response.text[: consts.WEB_FETCH_MAX_CHARS]
        text = html_to_markdown(body) if consts.WEB_HTML_MARK in kind or not kind else body
        return text_output(
            text,
            metadata={consts.META_URL: str(response.url), consts.META_VIA: consts.WEB_VIA_FETCH},
        )

    async def search(self, query: str, limit: int) -> ToolOutput:
        """Return titles, URLs and descriptions of the results."""
        key = self._key()
        if key is None:
            return error_output(consts.MSG_WEB_SEARCH_NEEDS_KEY, ErrorCode.TOOL_NOT_AVAILABLE)
        data = await self._firecrawl(consts.FIRECRAWL_SEARCH, key, {"query": query, "limit": limit})
        found = data.get("web", []) if isinstance(data, dict) else data
        rows = [row for row in found if isinstance(row, dict)] if isinstance(found, list) else []
        lines = [
            consts.WEB_RESULT_LINE.format(
                title=row.get("title") or "",
                url=row.get("url") or "",
                description=row.get("description") or "",
            )
            for row in rows
        ]
        return text_output(
            "\n\n".join(lines) if lines else consts.MSG_NO_MATCHES,
            metadata={consts.META_MATCHES: len(lines)},
        )


def _checked(url: str) -> str | None:
    parts = urlsplit(url)
    return None if parts.scheme in consts.WEB_SCHEMES and parts.netloc else url


def web_tools(
    config: ConfigManager, transport: httpx.AsyncBaseTransport | None = None
) -> list[FunctionTool]:
    """Build the web tools; ``transport`` replaces the network in tests."""
    web = WebClient(config, transport)

    @tool(
        group=consts.GROUP_WEB,
        read_only=True,
        reason_required=True,
        risk=RiskLevel.MEDIUM,
        default_permission=Decision.ASK,
        timeout_s=consts.WEB_TOOL_TIMEOUT_S,
        source=ToolSource.BUILTIN,
        traits=ToolTraits(match_param="url"),
    )
    async def web_scrape(url: str, formats: list[str] | None = None) -> ToolOutput:
        """Fetch a web page and return it as clean Markdown.

        ``formats`` picks what the web provider returns (default: ``["markdown"]``); it is
        ignored when the page is fetched directly.
        """
        if _checked(url) is not None:
            return error_output(consts.MSG_WEB_BAD_URL.format(url=url))
        try:
            return await web.scrape(url, formats or [consts.WEB_FORMAT_DEFAULT])
        except httpx.HTTPError as error:
            return error_output(consts.MSG_WEB_FAILED.format(url=url, error=error))

    @tool(
        group=consts.GROUP_WEB,
        read_only=True,
        reason_required=True,
        risk=RiskLevel.MEDIUM,
        default_permission=Decision.ASK,
        timeout_s=consts.WEB_TOOL_TIMEOUT_S,
        source=ToolSource.BUILTIN,
        traits=ToolTraits(match_param="query"),
    )
    async def web_search(query: str, limit: int = consts.WEB_SEARCH_LIMIT) -> ToolOutput:
        """Search the web and return titles, URLs and short descriptions."""
        try:
            return await web.search(query, max(1, min(limit, consts.WEB_SEARCH_LIMIT_MAX)))
        except httpx.HTTPError as error:
            return error_output(consts.MSG_WEB_FAILED.format(url=query, error=error))

    return [web_scrape, web_search]
