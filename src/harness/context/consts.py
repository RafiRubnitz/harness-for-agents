"""Context manager constants."""

from typing import Final

THINKING_BUDGET_TOKENS_DEFAULT: Final = 8_000
BYTES_PER_TOKEN_ESTIMATE: Final = 4
TEXT_ENCODING: Final = "utf-8"
SECTION_SEPARATOR: Final = "\n\n"
LINE_SEPARATOR: Final = "\n"

# --- System sections ---
BASE_INSTRUCTIONS: Final = (
    "You are an agent running inside a local harness on the user's own computer.\n"
    "- Tools: call a tool only when it moves the task forward, and say why in its "
    "'reason' field. A tool error is information: read it and adapt.\n"
    "- Long tool results are cut in the middle. The note left in the cut names a result "
    "id; read the missing part with read_tool_result(result_id, offset, limit).\n"
    "- Old tool results are replaced by a short note once you have moved on.\n"
    "- A <conversation_summary> block replaces older messages that were compacted.\n"
    "- A <memory> block holds facts remembered from earlier work. Each one starts with "
    "the date it was observed: weigh how fresh it is, and prefer the newer fact when two "
    "contradict each other.\n"
    "- A <hook_context> block is added by the user's automation; treat it as context.\n"
    "- Answer in the language the user writes in."
)
PROFILE_HEADER: Final = "# User profile"
PINNED_HEADER: Final = "# Pinned memories"

# --- Tagged blocks inside user messages ---
TAG_MEMORY: Final = "memory"
TAG_HOOK_CONTEXT: Final = "hook_context"
TAG_SUMMARY: Final = "conversation_summary"
TAGGED_BLOCK: Final = "<{tag}>\n{body}\n</{tag}>"
SUMMARY_BLOCK: Final = '<{tag} messages="{from_seq}-{to_seq}">\n{body}\n</{tag}>'
MEMORY_ITEM: Final = "[{date}] {content}"
MEMORY_DATE_FORMAT: Final = "%Y-%m-%d"

# --- Tool results at build time ---
TRUNCATE_HEAD_SHARE: Final = 0.5
MSG_TRUNCATED: Final = (
    "\n[... cut here: about {omitted} tokens left out. The whole result is stored as "
    "{result_id}; read it in parts with read_tool_result(result_id, offset, limit) ...]\n"
)
MSG_TRUNCATED_UNSTORED: Final = "\n[... cut here: about {omitted} tokens left out ...]\n"
OLD_RESULT_MIN_TOKENS: Final = 200
MSG_OLD_RESULT: Final = "[old tool result removed from the context (about {tokens} tokens)]"
MSG_OLD_RESULT_STORED: Final = (
    "[old tool result removed from the context (about {tokens} tokens); it is stored as "
    "{result_id}, read it with read_tool_result]"
)
MSG_OLD_IMAGE: Final = "[an image that was here was removed from the context]"

# --- Budget ---
EXACT_COUNT_FROM_SHARE: Final = 0.75
MAX_COMPACTIONS_PER_BUILD: Final = 2
MSG_PINNED_TRIMMED: Final = (
    "{dropped} pinned memories were left out of the context: the system part is over "
    "its budget of {budget} tokens"
)

# --- Compaction ---
SUMMARY_SYSTEM_PROMPT: Final = (
    "You compress the older part of a conversation between a user and an AI agent, so "
    "the agent can go on working without the original messages.\n"
    "Write a dense summary that keeps: the user's goals, decisions that were made and "
    "why, files that were created or changed (with paths), the current state of the "
    "work, open tasks and next steps, and facts the user stated about themselves or "
    "their preferences.\n"
    "Leave out pleasantries and tool output that no longer matters. Do not invent "
    "anything. Write in the main language of the conversation. Answer with the summary "
    "only."
)
SUMMARY_PREVIOUS_HEADER: Final = "# Summary so far"
SUMMARY_TRANSCRIPT_HEADER: Final = "# Messages to fold into the summary"
SUMMARY_INSTRUCTIONS_HEADER: Final = "# Extra instructions"
SUMMARY_MAX_OUTPUT_TOKENS: Final = 4_000
SUMMARY_CHUNK_WINDOW_SHARE: Final = 0.5
TRANSCRIPT_LINE: Final = "[{seq}] {role}: {text}"
TRANSCRIPT_BLOCK_MAX_CHARS: Final = 2_000
TRANSCRIPT_CUT_MARK: Final = " [...] "
TRANSCRIPT_TOOL_CALL: Final = "(calls tool {name} with {input})"
TRANSCRIPT_TOOL_RESULT: Final = "(tool result: {text})"
TRANSCRIPT_TOOL_ERROR: Final = "(tool error: {text})"
TRANSCRIPT_IMAGE: Final = "(image)"
TRANSCRIPT_FILE: Final = "(file)"

MSG_NOTHING_TO_COMPACT: Final = "there is nothing to compact yet"
MSG_COMPACTION_DENIED: Final = "compaction was blocked by a hook: {reasons}"
MSG_COMPACTION_FAILED: Final = "the summary model failed: {error}"
MSG_EMPTY_SUMMARY: Final = "the summary model returned no text"
REASON_JOINER: Final = "; "
