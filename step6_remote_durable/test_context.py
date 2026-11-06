"""Tests for memory, caching and compaction, with a scripted fake model."""
import copy
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock, Usage

import context
from agent import Agent, Workspace
from context import Meter, extract_summary, load_memory, with_summary_request
from permissions import Permissions
from sandbox import NoSandbox


def text(t):
    return TextBlock(type="text", text=t)


def tool(id, name, **input):
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input)


def reply(stop_reason, *content, tokens=100):
    usage = Usage(input_tokens=tokens, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage)


class FakeClient:
    def __init__(self, *responses):
        self.responses, self.requests, self.messages = list(responses), [], self

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "USER_MEMORY", tmp_path / "no-user-memory.md")
    return Workspace(tmp_path, NoSandbox())


# --- memory ---------------------------------------------------------------------------------

def test_memory_files_are_loaded_user_then_project(ws, tmp_path, monkeypatch):
    user = tmp_path / "user.md"
    user.write_text("I prefer uv over pip.")
    monkeypatch.setattr(context, "USER_MEMORY", user)
    (ws.root / "HARNESS.md").write_text("Run tests with: pytest -q")
    memory = load_memory(ws.root)
    assert memory.index("uv over pip") < memory.index("pytest -q")
    assert "pytest -q" in Agent(FakeClient(), ws).system[0]["text"]


def test_no_memory_file_is_fine(ws):
    assert load_memory(ws.root) == ""
    assert "Notes from earlier sessions" not in Agent(FakeClient(), ws).system[0]["text"]


def test_remember_appends_under_learned(ws):
    context.remember(ws.root, "tests live in tests/")
    context.remember(ws.root, "use ruff, not black")
    text = (ws.root / "HARNESS.md").read_text()
    assert text.count("## Learned") == 1
    assert "- tests live in tests/\n- use ruff, not black" in text


def test_remember_needs_approval_and_harness_md_cant_be_overwritten(ws):
    p = Permissions.load(ws.root, mode="accept-edits")
    assert p.check("remember", {"note": "x"}).action == "ask"           # even in accept-edits
    assert p.check("write_file", {"path": "HARNESS.md", "content": "x"}).action == "deny"


def test_memory_written_mid_session_does_not_change_this_sessions_prompt(ws):
    client = FakeClient(reply("tool_use", tool("t1", "remember", note="db is postgres")), reply("end_turn", text("ok")))
    agent = Agent(client, ws, permissions=Permissions(ask_user=lambda *a: ("yes", "")))
    agent.send("remember the db")
    assert "db is postgres" in (ws.root / "HARNESS.md").read_text()
    assert client.requests[0]["system"] == client.requests[1]["system"]


# --- caching --------------------------------------------------------------------------------

def test_requests_are_cache_friendly(ws):
    client = FakeClient(reply("tool_use", tool("t1", "run_bash", command="echo hi")), reply("end_turn", text("ok")),
                        reply("end_turn", text("again")))
    agent = Agent(client, ws, permissions=Permissions("bypass"))
    agent.send("one")
    agent.send("two")
    first, second, third = client.requests
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}   # breakpoint 1: tools + system
    assert first["cache_control"] == {"type": "ephemeral"}                # breakpoint 2: the conversation
    # Append-only: each request starts with the previous request's messages, unchanged.
    for before, after in [(first, second), (second, third)]:
        assert after["messages"][:len(before["messages"])] == before["messages"]
        assert after["system"] == before["system"] and after["tools"] == before["tools"]


def test_meter_math():
    m = Meter()
    m.add(Usage(input_tokens=1000, cache_creation_input_tokens=10_000, cache_read_input_tokens=0, output_tokens=500))
    m.add(Usage(input_tokens=200, cache_creation_input_tokens=1_000, cache_read_input_tokens=10_000, output_tokens=300))
    assert (m.input, m.cache_write, m.cache_read, m.output, m.requests) == (1200, 11_000, 10_000, 800, 2)
    assert m.last_context == 200 + 1_000 + 10_000 + 300
    assert m.cost == pytest.approx((1200 * 4 + 11_000 * 5 + 10_000 * 0.2 + 800 * 20) / 1e6)
    assert m.cost < m.cost_without_cache
    assert "45% from cache" in m.summary()   # 10,000 of 22,200 input tokens


# --- compaction -----------------------------------------------------------------------------

def test_summary_request_extends_the_conversation_without_changing_it():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [text("hello")]}]
    req = with_summary_request(history)
    assert req[:2] == history and req[2]["role"] == "user"
    assert "<summary>" in req[2]["content"][0]["text"]
    mid_task = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
    req = with_summary_request(mid_task)
    assert len(req) == 1 and req[0]["content"][0]["type"] == "tool_result"   # joined the same user turn
    assert len(mid_task[0]["content"]) == 1                                   # original untouched


def test_extract_summary():
    assert extract_summary([text("ok <summary>\nthe gist\n</summary> done")]) == "the gist"
    assert extract_summary([text("no tags")]) == "no tags"


def test_compaction_between_turns(ws):
    thinking = ThinkingBlock(type="thinking", thinking="", signature="sig")
    client = FakeClient(
        reply("end_turn", thinking, text("built the parser"), tokens=5000),     # first send: big context
        reply("end_turn", text("<summary>User wants a CSV parser. Done: parser.py.</summary>")),
        reply("end_turn", text("adding tests now")),
    )
    logs = []
    agent = Agent(client, ws, compact_at=1000, log=logs.append)
    agent.send("build a csv parser")
    assert agent.send("now add tests") == "adding tests now"

    summary_req, after = client.requests[1], client.requests[2]
    # The summary request is the conversation so far, unchanged (a cache hit), plus the
    # instruction added to the newest user turn.
    first_user, assistant, newest = summary_req["messages"]
    assert first_user == client.requests[0]["messages"][0] and assistant["role"] == "assistant"
    assert newest["content"][0]["text"] == "now add tests"
    assert "<summary>" in newest["content"][-1]["text"]
    # After compaction: one user turn with the summary and the new message, nothing replayed.
    [only] = after["messages"]
    assert only["role"] == "user"
    joined = " ".join(b["text"] for b in only["content"])
    assert "CSV parser" in joined and "now add tests" in joined
    assert agent.compactions == 1 and any("compacted" in line for line in logs)


def test_compaction_mid_task_carries_the_last_tool_result(ws):
    client = FakeClient(
        reply("tool_use", tool("t1", "run_bash", command="echo step-one-output"), tokens=5000),
        reply("end_turn", text("<summary>Running the migration; step one done.</summary>")),
        reply("end_turn", text("finished")),
    )
    agent = Agent(client, ws, permissions=Permissions("bypass"), compact_at=1000)
    assert agent.send("run the migration") == "finished"
    [only] = client.requests[2]["messages"]
    joined = " ".join(b["text"] for b in only["content"])
    assert "step one done" in joined and "step-one-output" in joined and "Continue the task" in joined
    assert not any(b.get("type") == "tool_result" for b in only["content"])   # no orphaned tool_result


def test_failed_summary_keeps_history(ws):
    client = FakeClient(reply("end_turn", text("first"), tokens=5000),
                        reply("tool_use", tool("t9", "read_file", path="x")),   # ignored the instruction
                        reply("end_turn", text("second")))
    agent = Agent(client, ws, compact_at=1000, log=lambda *_: None)
    agent.send("a")
    agent.send("b")
    assert agent.compactions == 0
    assert len(client.requests[2]["messages"]) == 3
