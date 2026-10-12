"""Tests for core/utils: ids, time, tool names, slash parsing, paths."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from harness.core.commands import AgentSend, ContextCompact, ContextInspect
from harness.core.consts import (
    ID_PREFIX_AGENT,
    ID_PREFIX_RUN,
    SLASH_CONTEXT_FULL_ARG,
    TOOL_WIRE_MAX_LENGTH,
    ErrorCode,
    SlashCommand,
)
from harness.core.errors import SlashError
from harness.core.types import TextBlock
from harness.core.utils.ids import UlidGenerator, is_valid_id, require_id
from harness.core.utils.paths import HarnessHome, is_within, resolve_home
from harness.core.utils.slash import parse_input
from harness.core.utils.time import SystemClock, elapsed_ms, ensure_utc
from harness.core.utils.tool_names import build_wire_map, from_wire_name, to_wire_name
from tests.consts import AGENT_ID, HEBREW_PLAIN

LONG_TOOL_NAME = "mcp.server." + "x" * (TOOL_WIRE_MAX_LENGTH * 2)


class TestIds:
    def test_generated_id_is_valid_for_its_prefix(self) -> None:
        new_id = UlidGenerator().new(ID_PREFIX_AGENT)

        assert is_valid_id(new_id, ID_PREFIX_AGENT)

    def test_generated_id_is_invalid_for_other_prefix(self) -> None:
        new_id = UlidGenerator().new(ID_PREFIX_AGENT)

        assert not is_valid_id(new_id, ID_PREFIX_RUN)

    def test_generated_ids_are_unique(self) -> None:
        generator = UlidGenerator()

        assert generator.new(ID_PREFIX_RUN) != generator.new(ID_PREFIX_RUN)

    def test_require_id_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match=ID_PREFIX_AGENT):
            require_id("not-an-id", ID_PREFIX_AGENT)


class TestTime:
    def test_system_clock_is_timezone_aware(self) -> None:
        assert SystemClock().now().tzinfo is not None

    def test_ensure_utc_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            ensure_utc(datetime(2026, 1, 1))

    def test_elapsed_ms_matches_timedelta(self) -> None:
        start = SystemClock().now()
        delta = timedelta(milliseconds=1500)

        assert elapsed_ms(start, start + delta) == int(delta.total_seconds() * 1000)


class TestToolNames:
    @pytest.mark.parametrize(
        "name",
        ["read_file", "mcp.github.create_issue", LONG_TOOL_NAME],
        ids=["plain", "mcp", "long"],
    )
    def test_wire_name_roundtrips_through_map(self, name: str) -> None:
        wire_map = build_wire_map([name])

        assert from_wire_name(to_wire_name(name), wire_map) == name

    def test_wire_name_respects_provider_limit(self) -> None:
        assert len(to_wire_name(LONG_TOOL_NAME)) <= TOOL_WIRE_MAX_LENGTH

    def test_distinct_long_names_stay_distinct(self) -> None:
        assert to_wire_name(LONG_TOOL_NAME + "a") != to_wire_name(LONG_TOOL_NAME + "b")

    def test_wire_name_has_no_dots(self) -> None:
        assert "." not in to_wire_name("mcp.github.create_issue")

    def test_unknown_wire_name_falls_back_to_separator_reversal(self) -> None:
        name = "mcp.github.create_issue"

        assert from_wire_name(to_wire_name(name), {}) == name


class TestSlash:
    def test_plain_text_becomes_send(self) -> None:
        command = parse_input(HEBREW_PLAIN, AGENT_ID)

        assert command == AgentSend(agent_id=AGENT_ID, content=[TextBlock(text=HEBREW_PLAIN)])

    def test_context_becomes_inspect(self) -> None:
        assert parse_input(SlashCommand.CONTEXT, AGENT_ID) == ContextInspect(agent_id=AGENT_ID)

    def test_context_full_sets_full(self) -> None:
        command = parse_input(f"{SlashCommand.CONTEXT} {SLASH_CONTEXT_FULL_ARG}", AGENT_ID)

        assert command == ContextInspect(agent_id=AGENT_ID, full=True)

    def test_compact_becomes_compact(self) -> None:
        assert parse_input(SlashCommand.COMPACT, AGENT_ID) == ContextCompact(agent_id=AGENT_ID)

    @pytest.mark.parametrize(
        "text",
        ["/contxt", SlashCommand.REFLECT, SlashCommand.PERMISSIONS],
        ids=["typo", "p2", "fut"],
    )
    def test_unknown_or_reserved_slash_is_not_sent(self, text: str) -> None:
        with pytest.raises(SlashError) as raised:
            parse_input(text, AGENT_ID)

        assert raised.value.code == ErrorCode.SLASH_UNKNOWN

    @pytest.mark.parametrize("text", ["/usr/bin/python fails", "/"], ids=["path", "bare-slash"])
    def test_path_like_text_is_sent_as_message(self, text: str) -> None:
        assert isinstance(parse_input(text, AGENT_ID), AgentSend)


class TestPaths:
    def test_resolve_home_prefers_override(self, tmp_path: Path) -> None:
        assert resolve_home(tmp_path).root == tmp_path.resolve()

    def test_resolve_home_reads_env(self, tmp_harness_home: HarnessHome) -> None:
        assert resolve_home().root == tmp_harness_home.root.resolve()

    def test_ensure_creates_directories(self, tmp_harness_home: HarnessHome) -> None:
        assert tmp_harness_home.workspaces_dir.is_dir()

    def test_is_within_accepts_child_and_rejects_sibling(self, tmp_path: Path) -> None:
        parent = tmp_path / "a"

        assert is_within(parent / "b" / "c.txt", parent)
        assert not is_within(tmp_path / "ab" / "c.txt", parent)

    def test_is_within_rejects_traversal(self, tmp_path: Path) -> None:
        parent = tmp_path / "a"

        assert not is_within(parent / ".." / "secret.txt", parent)
