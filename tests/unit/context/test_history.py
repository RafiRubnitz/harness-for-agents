"""The pure history transformations: thinking, tool results, cut points, transcripts."""

import pytest

from harness.context import history
from harness.context.utils import estimate_tokens, to_messages
from harness.core.consts import MessageRole
from harness.core.records import MessageRecord
from harness.core.types import (
    ContentBlock,
    ImageBlock,
    Source,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from tests.consts import HEBREW_PLAIN
from tests.utils import make_message

U, A, T = MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL
RESULT_ID = "blob_0042"
LIMIT = 100
KEEP_STEPS = 3
LONG = "0123456789 " * 400
HEAD_MARK, TAIL_MARK = "START-OF-RESULT ", " END-OF-RESULT"
THOUGHT = "let me think"


def record(seq: int, role: MessageRole, *content: ContentBlock) -> MessageRecord:
    blocks = list(content) or [TextBlock(text=f"message {seq}")]
    return make_message(f"msg_{seq}", role=role, content=blocks).model_copy(update={"seq": seq})


def call(call_id: str) -> ToolCallBlock:
    return ToolCallBlock(id=call_id, name="lookup", input={})


def result(call_id: str, text: str = "ok", result_id: str | None = None) -> ToolResultBlock:
    return ToolResultBlock(call_id=call_id, content=[TextBlock(text=text)], result_id=result_id)


def steps(count: int, text: str = LONG) -> list[MessageRecord]:
    """A user message followed by ``count`` tool-use steps."""
    records = [record(1, U)]
    for number in range(count):
        seq = 2 + number * 2
        records.append(record(seq, A, call(f"c{number}")))
        records.append(record(seq + 1, T, result(f"c{number}", text)))
    return records


def result_texts(records: list[MessageRecord]) -> list[str]:
    return [
        inner.text
        for item in records
        for block in item.content
        if isinstance(block, ToolResultBlock)
        for inner in block.content
        if isinstance(inner, TextBlock)
    ]


def is_safe(records: list[MessageRecord], cut: int) -> bool:
    """No call before the cut is answered after it, and the kept part opens no result."""
    before = {b.id for r in records[:cut] for b in r.content if isinstance(b, ToolCallBlock)}
    answered = {
        b.call_id for r in records[:cut] for b in r.content if isinstance(b, ToolResultBlock)
    }
    opens_with_result = cut < len(records) and records[cut].role is T
    return before == answered and not opens_with_result


# --- token estimate ---


def test_hebrew_counts_more_tokens_than_english_of_the_same_length() -> None:
    english = "a" * len(HEBREW_PLAIN)

    assert estimate_tokens(HEBREW_PLAIN) > estimate_tokens(english)


# --- thinking ---


def test_thinking_survives_only_in_the_current_turn() -> None:
    thinking = ThinkingBlock(text=THOUGHT, signature="sig")
    records = [
        record(1, U),
        record(2, A, thinking, call("c1")),
        record(3, T, result("c1")),
        record(4, A, thinking, TextBlock(text="done")),
        record(5, U),
        record(6, A, thinking, call("c2")),
        record(7, T, result("c2")),
    ]

    stripped = history.strip_old_thinking(records)

    kept = [r.seq for r in stripped if any(isinstance(b, ThinkingBlock) for b in r.content)]
    assert kept == [records[-2].seq]
    assert [len(r.content) for r in stripped[:4]] == [1, 1, 1, 1]


def test_a_record_left_empty_by_stripping_is_dropped_from_the_request() -> None:
    records = [
        record(1, U),
        record(2, A, ThinkingBlock(text=THOUGHT)),
        record(3, A, TextBlock(text="answer")),
        record(4, U),
    ]

    messages = to_messages(history.strip_old_thinking(records))

    assert [m.role for m in messages] == ["user", "assistant", "user"]


# --- big results ---


def test_long_text_keeps_its_start_and_end_and_names_the_stored_result() -> None:
    text = HEAD_MARK + LONG + TAIL_MARK

    cut = history.truncate_text(text, LIMIT, RESULT_ID)

    assert cut.startswith(HEAD_MARK)
    assert cut.endswith(TAIL_MARK)
    assert RESULT_ID in cut
    assert estimate_tokens(cut) < estimate_tokens(text)
    assert str(estimate_tokens(text) - LIMIT) in cut


def test_short_text_is_left_alone() -> None:
    assert history.truncate_text(HEAD_MARK, LIMIT, RESULT_ID) == HEAD_MARK


def test_only_oversized_results_are_cut() -> None:
    records = [
        record(1, A, call("small"), call("big")),
        record(2, T, result("small", "tiny"), result("big", LONG, RESULT_ID)),
    ]

    small, big = result_texts(history.shrink_results(records, LIMIT))

    assert small == "tiny"
    assert RESULT_ID in big
    assert len(big) < len(LONG)
    assert result_texts(records)[1] == LONG  # the stored history is untouched


# --- old results ---


def test_results_older_than_the_recent_steps_become_notes() -> None:
    records = steps(KEEP_STEPS * 2)

    aged = result_texts(history.stub_old_results(records, KEEP_STEPS))

    old, recent = aged[:KEEP_STEPS], aged[KEEP_STEPS:]
    assert all(text != LONG and len(text) < len(LONG) for text in old)
    assert recent == [LONG] * KEEP_STEPS


def test_small_old_results_stay_as_they_are() -> None:
    records = steps(KEEP_STEPS * 2, text="ok")

    assert history.stub_old_results(records, KEEP_STEPS) == records


@pytest.mark.parametrize("extra", range(1, KEEP_STEPS))
def test_the_old_line_does_not_move_on_every_step(extra: int) -> None:
    """The prefix stays byte-identical between jumps, so the prompt cache keeps hitting."""
    base = history.stub_old_results(steps(KEEP_STEPS * 2), KEEP_STEPS)

    later = history.stub_old_results(steps(KEEP_STEPS * 2 + extra), KEEP_STEPS)

    assert later[: len(base)] == base


def test_old_images_become_a_note() -> None:
    image = ImageBlock(source=Source(kind="base64", media_type="image/png", data="AAAA"))
    records = [record(1, U, image), *steps(KEEP_STEPS * 2, text="ok")[1:]]

    aged = history.stub_old_results(records, KEEP_STEPS)

    assert not any(isinstance(block, ImageBlock) for block in aged[0].content)


def test_a_zero_keep_switches_ageing_off() -> None:
    records = steps(KEEP_STEPS * 2)

    assert history.stub_old_results(records, 0) == records


# --- cut points ---


@pytest.mark.acceptance("07-12.2")
def test_no_cut_ever_splits_a_tool_call_from_its_result() -> None:
    records = [
        record(1, U),
        record(2, A, call("a"), call("b")),
        record(3, T, result("a")),
        record(4, T, result("b")),
        record(5, A),
        record(6, U),
        record(7, A, call("c")),
        record(8, T, result("c")),
        record(9, A, call("d")),
    ]

    cuts = [history.safe_cut(records, target) for target in range(len(records) + 1)]

    assert all(cut is not None and is_safe(records, cut) for cut in cuts)
    every_safe = {cut for cut in range(1, len(records) + 1) if is_safe(records, cut)}
    assert {cut for cut in cuts if cut is not None} == every_safe
    assert len(every_safe) < len(records)  # some cuts really are unsafe here


def test_the_nearest_safe_cut_below_the_target_wins() -> None:
    records = steps(4)

    assert history.safe_cut(records, 4) == 3
    assert history.safe_cut(records, 5) == 5


def test_an_unanswered_call_at_the_start_leaves_no_cut() -> None:
    records = [record(1, A, call("a")), record(2, U)]

    assert history.safe_cut(records, 1) is None


def test_the_tail_is_the_newest_part_that_fits() -> None:
    records = [record(seq, U, TextBlock(text=LONG)) for seq in range(1, 11)]
    one = estimate_tokens(records[0].content[0].model_dump_json())

    assert history.tail_start(records, one * 3) == len(records) - 3
    assert history.tail_start(records, one * len(records)) == 0
    assert history.tail_start(records, 0) == len(records) - 1  # the last one always stays


# --- transcript ---


def test_the_transcript_names_roles_tools_and_clips_long_data() -> None:
    records = [
        record(1, U, TextBlock(text="please look it up")),
        record(2, A, ThinkingBlock(text=THOUGHT), call("a")),
        record(3, T, result("a", LONG * 10)),
        record(4, T, ToolResultBlock(call_id="b", content=[TextBlock(text="boom")], is_error=True)),
    ]

    lines = history.transcript(records)

    assert len(lines) == len(records)
    assert "user" in lines[0]
    assert "please look it up" in lines[0]
    assert "lookup" in lines[1]
    assert THOUGHT not in lines[1]
    assert len(lines[2]) < len(LONG * 10)
    assert "error" in lines[3]


def test_lines_are_grouped_into_chunks_under_the_limit() -> None:
    lines = [LONG] * 10
    one = estimate_tokens(LONG)

    chunks = history.chunked(lines, one * 3)

    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]
    assert history.chunked(lines, 1) == [[line] for line in lines]
