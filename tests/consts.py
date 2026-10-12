"""Test input data and values chosen by the tests themselves (never copies of code constants)."""

from harness.core.types import AgentId, RunId

AGENT_ID = AgentId("agt_0001")
OTHER_AGENT_ID = AgentId("agt_0002")
RUN_ID = RunId("run_0001")
AGENT_NAME = "tester"
USER_ID = "tester-user"
TEST_MAX_STEPS = 3
TEST_MODEL_ALIAS = "main"
TEST_MODEL_ID = "fake-model-1"
TEST_API_KEY = "test-key-not-a-secret"

HEBREW_PLAIN = "שלום עולם, זה טקסט בעברית"
HEBREW_MIXED = "הפונקציה build_context() מחזירה ModelRequest"
HEBREW_NIQQUD = "שָׁלוֹם עוֹלָם"
HEBREW_FINALS = "ארץ שמים מלך סוף אמן"
HEBREW_SAMPLES = (HEBREW_PLAIN, HEBREW_MIXED, HEBREW_NIQQUD, HEBREW_FINALS)
