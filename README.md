# harness

A local multi-agent harness: several LLM agents running side by side on your own
computer, each with its own conversation, tools, permissions, memory and hooks.
Built from the "Harness MVP Spec" artifact.

Windows 11, Python 3.12, [uv](https://docs.astral.sh/uv/).

## Run

```sh
uv sync
uv run harness                  # the GUI (Tkinter)
uv run harness chat             # one agent in the terminal
uv run harness chat --new --model main
uv run harness agents           # list agents and their status
```

Everything lives in `~/.harness/` (override with `HARNESS_HOME`): `config.yaml`,
`hooks.yaml`, `templates/`, `hooks/`, `tools/`, the SQLite database, blobs, logs.

### API keys

Keys are read from the OS keyring (service `harness`), never from files:

```sh
uv run python -c "import keyring; keyring.set_password('harness', 'anthropic', 'sk-...')"
```

Names: `anthropic`, `openai`, `deepseek`, and `firecrawl` (optional, for the web tools).
The GUI settings dialog and the `config.set_secret` command store them the same way; the
dialog shows which providers already have a key (masked, like `sk-...a3f`).

### Provider per agent

Each agent has its own model, and every model belongs to a provider. The agent dialog
asks for the provider first, then for one of its models. Out of the box:

| Provider | Models (alias -> id) |
|---|---|
| anthropic | `main` -> `claude-opus-5-5`, `fast` -> `claude-sonnet-5-5`, `cheap` -> `claude-haiku-5-5` |
| openai | `gpt` -> `gpt-5`, `gpt-mini` -> `gpt-5-mini` |
| deepseek | `deepseek` -> `deepseek-chat`, `deepseek-reasoner` -> `deepseek-reasoner` |

Add or change models under `models:` in `config.yaml` (`provider`, `id`, optional
`fallback`). In the terminal: `uv run harness chat --new --model deepseek`.

### Semantic memory

Lexical search works out of the box. Vector search needs the embeddings server
(BGE-M3, about 2.3 GB on first use):

```sh
uv sync --extra embedder
```

To use an embeddings server that already runs elsewhere, set its URL in the settings
dialog (tab "memory") or as `embedder.url` in `config.yaml`, then restart. Leave it empty
and the harness starts its own local server. The server must speak this harness's own
protocol (`uv run python -m harness.embedder_server`); it is not an OpenAI-style endpoint.

## What is inside

| Component | Package | What it does |
|---|---|---|
| Core | `harness.core` | Types, commands, events, records, errors; imports no component |
| Models | `harness.models` | One gateway over Anthropic, OpenAI and DeepSeek: streaming, retry, fallback, cost |
| GUI / CLI | `harness.gui`, `harness.cli` | Front ends; they talk to the core only through commands and events |
| Memory | `harness.memory` | SQLite history, blobs, semantic memory (LanceDB), embeddings server |
| Agents | `harness.agents` | The agent loop, runs, inbox, approvals, crash recovery |
| Hooks | `harness.hooks` | User hooks (`hooks.yaml`): Python functions or commands; allow / ask / deny |
| Config | `harness.config` | Layered YAML (global, project, agent), secrets, templates, hot reload |
| Context | `harness.context` | Builds each model request; compaction, stored tool results |
| Tools | `harness.tools` | Registry, permission pipeline, safe execution, background tasks |

`harness.app` is the composition root; `harness.dispatcher` turns commands into calls.

### Tools

- Built in: `files` (read, write, edit, grep, list, delete with backup), `shell`,
  `memory`, `web` (`web_scrape`, `web_search`), and the always-on `context` and `tasks`
  groups.
- Custom: drop a Python file into `~/.harness/tools/`; it is picked up without a restart.

  ```python
  from harness.tools import tool


  @tool(group="dev", read_only=True, default_permission="allow")
  async def count_lines(path: str) -> str:
      """Count the lines of a file."""
      ...
  ```

- MCP: servers under `mcp_servers:` in `config.yaml` (a stdio `command` or an HTTP
  `url`); their tools appear as `mcp.<server>.<tool>` and ask before running.
- Any tool call can carry `run_in_background: true`; the agent gets a task id at once and
  a `<background_task_done>` message when the task ends.

## Development

```sh
uv run pytest                                   # tests, coverage must stay >= 80 %
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports                             # the architecture contracts
uv run python scripts/acceptance_report.py      # spec acceptance criteria -> tests
```

Tests never touch the network or a real model: providers, the embedder and the web are
replaced by fakes or mock transports. There are no live tests yet, so nothing in this
repository has been run against a real provider.
