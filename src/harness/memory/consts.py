"""Constants of the memory component."""

from typing import Final

# --- SQLite engine (13 §5.3) ---
SQLITE_DRIVER: Final = "sqlite+aiosqlite"
PRAGMAS: Final = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
)
PRAGMA_DATA_VERSION: Final = "PRAGMA data_version"
PRAGMA_FOREIGN_KEYS_OFF: Final = "PRAGMA foreign_keys=OFF"
PRAGMA_FOREIGN_KEY_CHECK: Final = "PRAGMA foreign_key_check"

# --- Migrations (13 §5) ---
MIGRATIONS_DIRNAME: Final = "migrations"
ALEMBIC_FILE_TEMPLATE: Final = "%%(rev)s_%%(slug)s"
ALEMBIC_HEAD: Final = "head"
ALEMBIC_BASE: Final = "base"
ALEMBIC_VERSION_TABLE: Final = "alembic_version"
FTS_TABLE: Final = "messages_fts"
FTS_SHADOW_SUFFIXES: Final = ("", "_data", "_idx", "_docsize", "_config", "_content")
BACKUP_KEEP: Final = 5
BACKUP_NAME: Final = "harness-{stamp}-{rev}.db"
BACKUP_GLOB: Final = "harness-*.db"
BACKUP_STAMP_FORMAT: Final = "%Y%m%dT%H%M%SZ"
BACKUP_EMPTY_REV: Final = "empty"
SQL_VACUUM_INTO: Final = "VACUUM INTO :path"
SQL_CURRENT_REVISION: Final = "SELECT version_num FROM alembic_version"
SQL_HAS_VERSION_TABLE: Final = (
    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = :name"
)

# --- harness_meta keys (13 §2.13) ---
META_CREATED_AT: Final = "created_at"
META_LAST_APP_VERSION: Final = "last_app_version"

# --- History search (13 §2.4) ---
SQL_SEARCH_HISTORY: Final = (
    "SELECT m.id AS id FROM messages_fts "
    "JOIN messages AS m ON m.pk = messages_fts.rowid "
    "WHERE messages_fts MATCH :q AND m.agent_id = :agent_id "
    "ORDER BY bm25(messages_fts) LIMIT :k"
)
HISTORY_SEARCH_LIMIT_DEFAULT: Final = 20
FTS_OR: Final = " OR "

# --- Hebrew normalisation (03 §6.3) ---
HEBREW_PREFIX_LETTERS: Final = "והבלמשכ"
HEBREW_MIN_STEM_LETTERS: Final = 3
HEBREW_MAX_PREFIX_LETTERS: Final = 3
HEBREW_FINALS: Final = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
HEBREW_MARKS_FIRST: Final = 0x0591
HEBREW_MARKS_LAST: Final = 0x05C7
HEBREW_PUNCTUATION_KEPT: Final = frozenset({0x05BE, 0x05C0, 0x05C3, 0x05C6})
HEBREW_LETTER_FIRST: Final = "א"
HEBREW_LETTER_LAST: Final = "ת"

# --- Blob store (13 §1) ---
BLOB_DELETED_DIRNAME: Final = "deleted"
BLOB_TASKS_DIRNAME: Final = "tasks"
BLOB_TASK_SUFFIX: Final = ".log"
BLOB_HASH_PREFIX_LENGTH: Final = 2
BLOB_DEFAULT_FILENAME: Final = "file"
DELETED_RETENTION_DAYS: Final = 30
META_AGENT_ID: Final = "agent_id"
META_FILENAME: Final = "filename"
META_MIME: Final = "mime"
META_TASK_ID: Final = "task_id"

# --- Messages ---
MSG_NAIVE_DATETIME: Final = "naive datetime cannot be stored; use a timezone-aware UTC value"
MSG_AGENT_NOT_FOUND: Final = "agent {agent_id} does not exist"
MSG_BLOB_NOT_FOUND: Final = "blob {blob_id} does not exist"
MSG_BACKUP_NOT_FOUND: Final = "deletion backup {backup_id} does not exist"
MSG_BLOB_IN_USE: Final = "blob {blob_id} is still referenced"
MSG_FK_VIOLATIONS: Final = "foreign key check failed after migration: {rows}"
MSG_DB_FAILED: Final = "database operation failed: {error}"
MSG_TASK_ID_REQUIRED: Final = "a task_output blob needs meta['task_id']"
MSG_TOOL_CALL_NOT_FOUND: Final = "no tool call {call_id} for agent {agent_id}"
MSG_OPEN_RUN_EXISTS: Final = "agent {agent_id} already has an open run {run_id}"

# --- Semantic memory: LanceDB (13 §4) ---
EMBED_DIM: Final = 1024
SEMANTIC_SCHEMA_VERSION: Final = 1
SCHEMA_VERSION_KEY: Final = b"harness.schema_version"
TABLE_MEMORIES: Final = "memories"
TABLE_VERSIONS: Final = "memory_versions"
COL_ID: Final = "id"
COL_ITEM_ID: Final = "item_id"
COL_AGENT_ID: Final = "agent_id"
COL_SCOPE: Final = "scope"
COL_KIND: Final = "kind"
COL_CONTENT_NORM: Final = "content_norm"
COL_VECTOR: Final = "vector"
COL_SCHEMA_VERSION: Final = "schema_version"
COL_VERSION_NO: Final = "version_no"
COL_ACCESS_COUNT: Final = "access_count"
COL_LAST_ACCESSED_AT: Final = "last_accessed_at"
COL_DISTANCE: Final = "_distance"
COL_SCORE: Final = "_score"
BAD_VECTOR_AS_NULL: Final = "null"  # a missing vector is stored as NULL, not refused
SQL_ACCESS_INCREMENT: Final = "access_count + 1"
DISTANCE_COSINE: Final = "cosine"
FTS_BASE_TOKENIZER: Final = "whitespace"
BTREE_COLUMNS: Final = (COL_ID, COL_AGENT_ID)
BITMAP_COLUMNS: Final = (COL_SCOPE, COL_KIND)
OPTIMIZE_EVERY_WRITES: Final = 200
SAFE_ID_PATTERN: Final = r"[A-Za-z0-9_\-]{1,80}"
SAFE_MODEL_PATTERN: Final = r"[A-Za-z0-9_\-./:]{1,200}"

# --- Semantic memory: search and upkeep (03 §6) ---
SEARCH_CANDIDATES: Final = 50
RRF_K: Final = 60
RECENCY_HALF_LIFE_DAYS: Final = 30.0
SECONDS_PER_DAY: Final = 86_400.0
FILTER_KIND: Final = "kind"
FILTER_TAGS: Final = "tags"
CHANGE_KIND: Final = "kind"
CHANGE_TAGS: Final = "tags"
CHANGE_PINNED: Final = "pinned"
CHANGE_OBSERVED_AT: Final = "observed_at"
ALLOWED_CHANGES: Final = frozenset({CHANGE_KIND, CHANGE_TAGS, CHANGE_PINNED, CHANGE_OBSERVED_AT})
BACKFILL_BATCH: Final = 16
BACKFILL_INTERVAL_S: Final = 30.0
OP_REMEMBER: Final = "remember"
OP_UPDATE: Final = "update"
OP_SUPERSEDE: Final = "supersede"
PAYLOAD_CONTENT: Final = "content"
PAYLOAD_TAGS: Final = "tags"
GLOBAL_AGENT_NAME: Final = ""

# --- Embeddings client and managed server (03 §6.4) ---
EMBED_KIND_DOCUMENT: Final = "document"
EMBED_KIND_QUERY: Final = "query"
HEALTH_TIMEOUT_S: Final = 2.0
EMBED_HTTP_TIMEOUT_S: Final = 300.0
EMBEDDER_URL: Final = "http://{host}:{port}"
EMBEDDER_WATCH_INTERVAL_S: Final = 5.0
EMBEDDER_STOP_TIMEOUT_S: Final = 5.0
EMBEDDER_MAX_RESTARTS: Final = 3
EMBEDDER_LOG_NAME: Final = "embedder.log"

MSG_SEMANTIC_NEWER: Final = (
    "semantic memory was written by a newer version of the app "
    "(schema {found}, this build understands {known}); it was not opened"
)
MSG_SEMANTIC_NO_MIGRATION: Final = "no migration from semantic schema {found} to {known}"
MSG_SEMANTIC_FAILED: Final = "semantic memory operation failed: {error}"
MSG_UNSAFE_FILTER_VALUE: Final = "value {value!r} cannot be used in a memory filter"
MSG_ITEM_NOT_FOUND: Final = "memory item {item_id} does not exist"
MSG_UNKNOWN_CHANGE: Final = "memory items have no editable field {field!r}"
MSG_EMPTY_CONTENT: Final = "a memory item needs some content"
MSG_WRITE_DENIED: Final = "a hook did not allow this memory write: {reason}"
MSG_WRITE_DENIED_NO_REASON: Final = "no reason given"
MSG_EMBEDDER_DOWN: Final = "the embeddings server is unavailable: {error}"
MSG_EMBEDDER_BAD_REPLY: Final = "the embeddings server sent {got} vectors for {wanted} texts"
MSG_EMBEDDER_FALLBACK: Final = (
    "the embeddings server is unavailable; memory search is lexical only for now"
)
MSG_REINDEX_MODEL_MISMATCH: Final = (
    "cannot reindex to {wanted}: the embeddings server runs {running}"
)
MSG_EMBEDDER_GAVE_UP: Final = "the embeddings server keeps exiting; it will not be restarted"
