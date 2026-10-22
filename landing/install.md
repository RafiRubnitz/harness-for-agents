# Install the harness end to end

You are an AI coding agent with a shell on the user's computer. Your job is to install
**harness** (a local multi-agent harness, https://github.com/RafiRubnitz/harness-for-agents)
from scratch, verify it works, and leave the user ready to launch it.

Work through the steps in order. After each step, run its check. If a check fails, fix
the cause before moving on and tell the user what you changed. Do not skip steps.

## Rules

- **Never ask the user to paste an API key into the chat, and never print, log or
  write a key to a file.** Keys go into the OS keyring only, and the user types them
  themselves (step 5).
- Do not install anything system-wide beyond what is listed here without asking.
- Target platform: Windows 11 (the harness is built and tested there). On macOS or
  Linux the same commands work, but tell the user the platform is untested.
- Use the shell the user already has. Commands below are shown for PowerShell; adapt
  quoting if you are in bash.

## 1. Prerequisites

Check what is already installed:

```sh
git --version
uv --version
```

- **git** missing: on Windows run `winget install --id Git.Git -e`, then open a new shell.
- **uv** missing: on Windows run
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`), then open a new shell.
- **Python 3.12** is required (`>=3.12,<3.13`). You do not need to install it yourself:
  uv reads `.python-version` in the repository and downloads 3.12 if it is missing.

Check: both `git --version` and `uv --version` print a version.

## 2. Get the code

Ask the user where to put the project if they have not said; default to their home
directory.

```sh
git clone https://github.com/RafiRubnitz/harness-for-agents
cd harness-for-agents
```

If the folder already exists, `cd` into it and run `git pull` instead of cloning again.

Check: `pyproject.toml` and `.python-version` exist in the current folder.

## 3. Install dependencies

```sh
uv sync
```

This creates `.venv/` with Python 3.12 and every dependency. It can take a few minutes
the first time.

Check:

```sh
uv run python --version          # must print Python 3.12.x
uv run harness --help            # must list the chat and agents commands
```

## 4. Run the test suite (sanity check)

```sh
uv run pytest -q
```

Tests use fakes only; they never touch the network or a real model, so they need no keys.

Check: the run ends with all tests passing. If some fail, report the failing test names
and the error to the user, then continue; a failing test does not block the install.

## 5. Store API keys (the user does this)

The harness reads keys from the OS keyring under the service name `harness`, never from
files. Supported key names:

| Name | Needed for |
|---|---|
| `anthropic` | Claude models (`main`, `fast`, `cheap`) |
| `openai` | `gpt`, `gpt-mini` |
| `deepseek` | `deepseek`, `deepseek-reasoner` |
| `firecrawl` | optional, the `web_scrape` / `web_search` tools |

At least one model provider key is required. Ask the user which providers they want,
then give them this command to run **themselves**, once per key, replacing the name and
the key:

```sh
uv run python -c "import keyring; keyring.set_password('harness', 'anthropic', 'PASTE-KEY-HERE')"
```

Alternatively they can enter keys later in the GUI settings dialog, which stores them the
same way.

Check (prints only whether each key exists, never the key itself):

```sh
uv run python -c "import keyring; [print(n, 'set' if keyring.get_password('harness', n) else 'missing') for n in ('anthropic','openai','deepseek','firecrawl')]"
```

## 6. First run

```sh
uv run harness agents
```

On first run the harness creates its home folder `~/.harness/` (override with the
`HARNESS_HOME` environment variable) with the SQLite database `harness.db` and folders
such as `templates/`, `hooks/`, `tools/`, `blobs/` and `logs/`. `config.yaml` and
`hooks.yaml` are not created up front; they appear when the user changes settings, or
can be written by hand in that folder.

Check: the command prints `no agents`, exits without an error, and `~/.harness/harness.db`
exists.

Then start a terminal chat with one agent, using a model whose provider has a key:

```sh
uv run harness chat --new --model main       # anthropic
uv run harness chat --new --model gpt        # openai
uv run harness chat --new --model deepseek   # deepseek
```

Send one short message and confirm a reply streams back, then exit the chat.

## 7. Optional: semantic memory

Lexical memory search works out of the box. For vector search, install the embeddings
extra (downloads the BGE-M3 model, about 2.3 GB, on first use). Ask the user before doing
this because of the download size.

```sh
uv sync --extra embedder
```

If the user already runs a compatible embeddings server, set its URL as `embedder.url` in
`~/.harness/config.yaml` (or in the GUI settings, tab "memory") instead. It must speak this
harness's own protocol (`uv run python -m harness.embedder_server`), not an OpenAI-style API.

## 8. Hand over

Tell the user:

- how to launch the desktop GUI: `uv run harness` (from the project folder),
- how to chat in the terminal: `uv run harness chat`,
- where config lives: `~/.harness/config.yaml`; custom tools go in `~/.harness/tools/`,
  hooks in `~/.harness/hooks.yaml`, MCP servers under `mcp_servers:` in `config.yaml`,
- which keys are set and which are missing (from the step 5 check),
- any step that failed or was skipped, with the error.

Note for the user: this is an MVP (v0.1.0) and has not yet been tested against live
providers, so the first real conversation may surface rough edges.
