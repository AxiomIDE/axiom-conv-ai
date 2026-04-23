"""Unit tests for MemoryChat.

The node is async and depends on ax.agent.memory and the Anthropic API.
Both are replaced with fakes so no real I/O happens during tests.
"""
import asyncio
from unittest.mock import MagicMock, patch

from gen.messages_pb2 import ConvRequest, ConvResponse
from nodes.memory_chat import memory_chat


# ---------------------------------------------------------------------------
# Fake memory hierarchy
# ---------------------------------------------------------------------------

class _FakeSessionHistory:
    def __init__(self, turns=None):
        self._turns = turns or []
        self.appended = []

    async def last(self, n: int):
        return self._turns[-n:] if n else self._turns

    async def append(self, *, role: str, content: str):
        self.appended.append({"role": role, "content": content})


class _FakeSessionMemory:
    def __init__(self, turns=None, semantic_hits=None):
        self.history = _FakeSessionHistory(turns)
        self._semantic_hits = semantic_hits or []
        self.written = []

    async def search(self, query: str, limit: int = 5):
        return self._semantic_hits[:limit]

    async def write(self, content: str, importance: float = 0.5):
        self.written.append({"content": content, "importance": importance})
        return "mem-id-1"

    async def end(self):
        pass


class _FakeAgentMemory:
    def __init__(self, session: _FakeSessionMemory):
        self._session = session

    def session(self, session_id: str):
        return self._session

    async def search(self, query: str, limit: int = 5):
        return []

    async def write(self, content: str, importance: float = 0.5):
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
        self.execution_id = "exec-001"
        self.flow_id = "flow-001"
        self.tenant_id = "tenant-001"


# ---------------------------------------------------------------------------
# Helper: build a fake Anthropic response
# ---------------------------------------------------------------------------

def _mock_anthropic_response(text: str):
    content_block = MagicMock()
    content_block.text = text
    resp = MagicMock()
    resp.content = [content_block]
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memory_chat_returns_llm_response():
    """The node must return the LLM reply as ConvResponse.response."""
    session = _FakeSessionMemory()
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s1", user_message="Hello!")

    with patch("nodes.memory_chat.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = _mock_anthropic_response("Hi there!")
        result = asyncio.run(memory_chat(ax, req))

    assert isinstance(result, ConvResponse)
    assert result.session_id == "s1"
    assert result.response == "Hi there!"


def test_memory_chat_appends_both_turns_to_history():
    """Both user and assistant turns must be written to session history."""
    session = _FakeSessionMemory()
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s2", user_message="What is Axiom?")

    with patch("nodes.memory_chat.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = _mock_anthropic_response("Axiom is a platform.")
        asyncio.run(memory_chat(ax, req))

    roles = [t["role"] for t in session.history.appended]
    contents = [t["content"] for t in session.history.appended]
    assert "user" in roles
    assert "assistant" in roles
    assert "What is Axiom?" in contents
    assert "Axiom is a platform." in contents


def test_memory_chat_writes_semantic_memory():
    """The node must write at least one entry to semantic memory."""
    session = _FakeSessionMemory()
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s3", user_message="Remember this fact.")

    with patch("nodes.memory_chat.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.return_value = _mock_anthropic_response("Noted.")
        asyncio.run(memory_chat(ax, req))

    assert len(session.written) >= 1
    assert session.written[0]["importance"] > 0


def test_memory_chat_incorporates_prior_history():
    """Prior session turns must be included in the LLM call messages."""
    prior_turn = MagicMock()
    prior_turn.role = "user"
    prior_turn.content = "My name is Alice."
    session = _FakeSessionMemory(turns=[prior_turn])
    ax = _FakeContext(session)
    req = ConvRequest(session_id="s4", user_message="What is my name?")

    captured_messages = []

    def _capture(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return _mock_anthropic_response("Your name is Alice.")

    with patch("nodes.memory_chat.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value.messages.create.side_effect = _capture
        asyncio.run(memory_chat(ax, req))

    contents = [m["content"] for m in captured_messages]
    assert any("Alice" in c for c in contents), "Prior turn content must reach the LLM"


def test_memory_chat_missing_secret_raises():
    """RuntimeError must be raised when ANTHROPIC_API_KEY is absent."""
    session = _FakeSessionMemory()
    ax = _FakeContext(session, secrets={})
    req = ConvRequest(session_id="s5", user_message="Hi")

    try:
        asyncio.run(memory_chat(ax, req))
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
