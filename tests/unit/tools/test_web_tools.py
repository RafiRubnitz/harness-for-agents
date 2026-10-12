"""The web tools: Firecrawl with a key, a plain fetch without one (08 §4.4)."""

import asyncio
import json

import httpx
import pytest
from pydantic import JsonValue

from harness.core.consts import Decision, ErrorCode
from harness.core.records import ToolOutput
from harness.tools.builtin.markdown import html_to_markdown
from harness.tools.builtin.web import web_tools
from harness.tools.consts import GROUP_WEB, META_MATCHES, META_VIA
from harness.tools.types import ToolContext
from tests.consts import AGENT_ID
from tests.unit.tools.conftest import ToolsEnv, text_of

KEY = "fc-test-key-not-a-secret"
SECRET = "firecrawl"
PAGE_URL = "https://example.test/guide"
PAGE_HOST = "example.test"
PROVIDER_HOST = "api.firecrawl.dev"
MARKDOWN = "# Guide\n\nRead the *manual* first."
TITLE = "Install guide"
BODY_TEXT = "Read the manual first."
LINK_TEXT = "the manual"
LINK_URL = "https://example.test/manual"
HIDDEN = "tracking()"
HTML = (
    f"<html><head><title>x</title><script>{HIDDEN}</script></head><body>"
    f"<h1>{TITLE}</h1><p>{BODY_TEXT} See <a href='{LINK_URL}'>{LINK_TEXT}</a>.</p>"
    "<ul><li>first</li><li>second</li></ul></body></html>"
)
QUERY = "harness install guide"
RESULTS: list[dict[str, JsonValue]] = [
    {"title": TITLE, "url": PAGE_URL, "description": BODY_TEXT},
    {"title": "Manual", "url": LINK_URL, "description": "The whole manual."},
]
SERVER_ERROR = 503


class Net:
    """Stands in for the network and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status: int = httpx.codes.OK

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != httpx.codes.OK:
            return httpx.Response(self.status)
        if request.url.host != PROVIDER_HOST:
            return httpx.Response(httpx.codes.OK, html=HTML)
        if request.url.path.endswith("/search"):
            return httpx.Response(httpx.codes.OK, json={"success": True, "data": {"web": RESULTS}})
        return httpx.Response(
            httpx.codes.OK, json={"success": True, "data": {"markdown": MARKDOWN}}
        )

    def hosts(self) -> list[str]:
        return [request.url.host for request in self.requests]


class Web:
    """Calls the web tools the way the executor does."""

    def __init__(self, env: ToolsEnv) -> None:
        self.env = env
        self.net = Net()
        tools = web_tools(env.config, httpx.MockTransport(self.net))
        self.tools = {tool.definition.name: tool for tool in tools}
        self.ctx = ToolContext(
            agent_id=AGENT_ID,
            run_id=None,
            call_id="call_1",
            working_dir=str(env.work),
            allowed_paths=[],
            cancel=asyncio.Event(),
            emit_progress=lambda chunk: None,
            clock=env.clock,
        )

    async def call(self, name: str, **args: JsonValue) -> ToolOutput:
        return await self.tools[name].run(args, self.ctx)


@pytest.fixture
def web(env: ToolsEnv) -> Web:
    return Web(env)


@pytest.mark.acceptance("08-10.14")
async def test_scrape_returns_markdown_through_firecrawl_when_there_is_a_key(web: Web) -> None:
    await web.env.config.set_secret(SECRET, KEY)

    output = await web.call("web_scrape", url=PAGE_URL)

    (request,) = web.net.requests
    assert text_of(output) == MARKDOWN
    assert request.url.host == PROVIDER_HOST
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content)["url"] == PAGE_URL
    assert output.metadata[META_VIA] == SECRET


@pytest.mark.acceptance("08-10.14")
async def test_scrape_without_a_key_fetches_the_page_itself_as_markdown(web: Web) -> None:
    output = await web.call("web_scrape", url=PAGE_URL)

    text = text_of(output)
    assert web.net.hosts() == [PAGE_HOST]
    assert f"# {TITLE}" in text
    assert f"[{LINK_TEXT}]({LINK_URL})" in text
    assert "- first" in text
    assert HIDDEN not in text
    assert "<" not in text
    assert output.metadata[META_VIA] != SECRET


async def test_search_lists_titles_urls_and_descriptions(web: Web) -> None:
    await web.env.config.set_secret(SECRET, KEY)

    output = await web.call("web_search", query=QUERY)

    (request,) = web.net.requests
    assert json.loads(request.content)["query"] == QUERY
    assert output.metadata[META_MATCHES] == len(RESULTS)
    assert all(str(row[field]) in text_of(output) for row in RESULTS for field in row)


async def test_search_is_not_available_without_a_key(web: Web) -> None:
    output = await web.call("web_search", query=QUERY)

    assert output.is_error
    assert output.metadata["error_code"] == ErrorCode.TOOL_NOT_AVAILABLE.value
    assert web.net.requests == []


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.test/x", "example.test"])
async def test_only_http_urls_are_fetched(web: Web, url: str) -> None:
    output = await web.call("web_scrape", url=url)

    assert output.is_error
    assert web.net.requests == []


@pytest.mark.parametrize("with_key", [True, False])
async def test_a_failing_request_is_an_error_the_model_can_read(web: Web, with_key: bool) -> None:
    if with_key:
        await web.env.config.set_secret(SECRET, KEY)
    web.net.status = SERVER_ERROR

    output = await web.call("web_scrape", url=PAGE_URL)

    assert output.is_error
    assert PAGE_URL in text_of(output)


def test_the_web_tools_ask_before_they_run(env: ToolsEnv) -> None:
    listed = [d for d in env.manager.list() if d.group == GROUP_WEB]

    assert {d.name for d in listed} == {"web_scrape", "web_search"}
    assert {d.default_permission for d in listed} == {Decision.ASK}


def test_markdown_keeps_structure_and_drops_markup() -> None:
    html = "<h2>Steps</h2><p>Run <code>make</code> <b>twice</b>.</p><pre>a\n  b</pre>"

    assert html_to_markdown(html) == "## Steps\n\nRun `make` **twice**.\n\n```\na\n  b\n```"
