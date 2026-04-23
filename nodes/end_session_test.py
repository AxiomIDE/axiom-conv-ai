"""Unit tests for EndSession.

The node depends on ax.agent.memory and the Anthropic API.
Both are replaced with fakes so no real I/O happens during tests.
"""
from unittest.mock import MagicMock, patch

from gen.messages_pb2 import ConvRequest, ConvResponse
from nodes.end_session import end_session


# ---------------------------------------------------------------------------
# Fake memory hierarchy (mirrors memory_chat_test.py)
# ---------------------------------------------------------------------------

class _FakeSessionHistory:
    def __init__(self, turns=None):
        self._turns = turns or []

    async def last(self, n: int):
        return self._turns[-n:] if n else self._turns


class _FakeSessionMemory:
    def __init__(self, turns=None):
        self.history = _FakeSessionHistory(turns)
        self.ended = False

    async def search(self, query: str, limit: int = 5):
        return []

    async def write(self, content: str, importance: float = 0.5):
        return "mem-id-session"

    async def end(self):
        self.ended = True


class _FakeAgentMemory:
    def __init__(self, session: _FakeSessionMemory):
        self._session = session
        self.global_written = []

    def session(self, session_id: str):
        return self._session

    async def search(self, query: str, limit: int = 5):
        return []

    async def write(self, content: str, importance: float = 0.5):
        self.global_written.append({"content": content, "importance": importance})
        return "mem-id-global"


class _FakeAgent:
    def __init__(self, session: _FakeSessionMemory):
        self.memory = _FakeAgentMemory(session)


class _FakeLogger:
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _FakeSecrets:
    def __init__(self, m: dict):
        self._m = m

    def get(self, name: str):
        v = self._m.get(name)
        return (v, True) if v is not None else ("", False)


class _FakeContext:
    def __init__(self, session: _FakeSessionMemory, secrets: dict | None = None):
        self.log = _FakeLogger()
        self.secrets = _FakeSecrets({"ANTHROPIC_API_KEY": "sk-test"} if secrets is None else secrets)
        self.agent = _FakeAgent(session)
        self.execution_id = "exec-002"
        self.flow_id = "flow-002"
        self.tenant_id = "tenant-001"


def _mock_response(text: str):
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_turn(role: str, content: str):
    t = MagicMock()
    t.role = role
    t.content = content
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_end_session_closes_session():
    """session.end() must be called to trigger consolidation."""
    turns = [_make_turn("user", "Hello"), _make_turn("assistant", "Hi")]
    session = _FakeSessionMemory(turns=turns)
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s1", user_message="")

    with patch("nodes.end_session.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = _mock_response("A short chat.")
        end_session(ax, req)

    assert session.ended, "session.end() must be called"


def test_end_session_writes_global_semantic_memory():
    """Summary must be written as a cross-session (flow-scoped) semantic memory."""
    turns = [_make_turn("user", "Tell me about Axiom"), _make_turn("assistant", "Axiom is a platform.")]
    session = _FakeSessionMemory(turns=turns)
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s2", user_message="")

    with patch("nodes.end_session.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = _mock_response("User asked about Axiom.")
        end_session(ax, req)

    assert len(ax.agent.memory.global_written) >= 1
    written = ax.agent.memory.global_written[0]
    assert written["importance"] >= 0.5
    assert len(written["content"]) > 0


def test_end_session_response_contains_summary():
    """The ConvResponse.response must include the generated summary."""
    turns = [_make_turn("user", "Hello"), _make_turn("assistant", "Hi!")]
    session = _FakeSessionMemory(turns=turns)
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s3", user_message="")

    with patch("nodes.end_session.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = _mock_response("A brief greeting exchange.")
        result = end_session(ax, req)

    assert isinstance(result, ConvResponse)
    assert result.session_id == "s3"
    assert "A brief greeting exchange." in result.response


def test_end_session_no_turns_closes_without_llm():
    """An empty session should close cleanly without calling the LLM."""
    session = _FakeSessionMemory(turns=[])
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s4", user_message="")

    with patch("nodes.end_session.anthropic.Anthropic") as mock_cls:
        result = end_session(ax, req)
        mock_cls.return_value.messages.create.assert_not_called()

    assert session.ended
    assert "No conversation history" in result.response


def test_end_session_missing_secret_raises():
    """RuntimeError must be raised when ANTHROPIC_API_KEY is absent."""
    session = _FakeSessionMemory(turns=[_make_turn("user", "hi")])
    ax = _FakeContext(session, secrets={})
    req = ConvRequest(session_id="s5", user_message="")

    try:
        end_session(ax, req)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
