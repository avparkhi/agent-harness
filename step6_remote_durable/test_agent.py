"""Loop tests carried over from step 1, plus how permissions plug into the loop.
A scripted fake client stands in for the API, so these run without an API key."""
import copy
import time
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock, Usage

import agent as agent_mod
from agent import Agent, Workspace
from permissions import Permissions
from sandbox import NoSandbox


def text(t):
    return TextBlock(type="text", text=t)


def tool(id, name, **input):
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input)


def reply(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content),
                           usage=Usage(input_tokens=10, output_tokens=5))


class FakeClient:
    """Returns scripted responses in order and records every request it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path, NoSandbox())


def Agent(client, ws, **kw):
    """Step 1's loop tests run with everything allowed; the permission tests are below."""
    import agent
    kw.setdefault("permissions", Permissions("bypass"))
    return agent.Agent(client, ws, **kw)


def test_plain_answer_ends_after_one_call(ws):
    client = FakeClient(reply("end_turn", text("hello")))
    assert Agent(client, ws).send("hi") == "hello"
    assert len(client.requests) == 1
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert [t["name"] for t in req["tools"]] == ["read_file", "write_file", "run_bash", "remember",
                                                "search_files", "todo_write", "task"]
    assert str(ws.root) in req["system"][0]["text"]


def test_tool_call_runs_and_result_goes_back_as_user_turn(ws):
    client = FakeClient(
        reply("tool_use", text("writing it"), tool("t1", "write_file", path="a/hi.txt", content="hey")),
        reply("end_turn", text("done")),
    )
    assert Agent(client, ws).send("make a file") == "done"
    assert (ws.root / "a/hi.txt").read_text() == "hey"

    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    [result] = second[2]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert result["is_error"] is False


def test_parallel_tool_calls_answered_in_one_user_turn(ws):
    (ws.root / "x.txt").write_text("X")
    client = FakeClient(
        reply("tool_use", tool("t1", "read_file", path="x.txt"), tool("t2", "run_bash", command="echo hi")),
        reply("end_turn", text("ok")),
    )
    Agent(client, ws).send("go")
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[0]["content"] == "X"
    assert "exit code: 0" in results[1]["content"] and "hi" in results[1]["content"]


def test_assistant_turn_is_appended_unchanged_with_thinking(ws):
    thinking = ThinkingBlock(type="thinking", thinking="", signature="sig")
    response = reply("end_turn", thinking, text("hi"))
    a = Agent(FakeClient(response), ws)
    a.send("hi")
    assert a.messages[1] == {"role": "assistant", "content": response.content}


def test_path_outside_workspace_is_an_error_not_a_crash(ws):
    client = FakeClient(
        reply("tool_use", tool("t1", "write_file", path="../escape.txt", content="x")),
        reply("end_turn", text("sorry")),
    )
    Agent(client, ws).send("go")
    [result] = client.requests[1]["messages"][2]["content"]
    assert result["is_error"] is True and "outside the working directory" in result["content"]
    assert not (ws.root.parent / "escape.txt").exists()


def test_missing_file_and_unknown_tool_are_errors(ws):
    assert ws.run("read_file", {"path": "nope.txt"})[1] is True
    assert ws.run("delete_everything", {}) == ("unknown tool: delete_everything", True)


def test_bash_timeout(ws, monkeypatch):
    monkeypatch.setattr(agent_mod, "BASH_TIMEOUT", 1)
    start = time.time()
    out, is_error = ws.run("run_bash", {"command": "sleep 5"})
    assert is_error and "timed out" in out
    assert time.time() - start < 4


def test_long_output_is_cut(ws, monkeypatch):
    monkeypatch.setattr(agent_mod, "MAX_RESULT_CHARS", 100)
    out, _ = ws.run("run_bash", {"command": "yes | head -c 1000"})
    assert len(out) < 200 and "[... cut" in out


def test_turn_cap_stops_a_runaway_loop(ws):
    looping = [reply("tool_use", tool(f"t{i}", "run_bash", command="true")) for i in range(3)]
    client = FakeClient(*looping)
    assert "cap" in Agent(client, ws, max_turns=3).send("go")
    assert len(client.requests) == 3


def test_cut_off_tool_call_gets_error_result_and_loop_continues(ws):
    client = FakeClient(
        reply("max_tokens", tool("t1", "write_file", path="big.txt", content="partial")),
        reply("end_turn", text("retrying smaller")),
    )
    assert Agent(client, ws).send("go") == "retrying smaller"
    [result] = client.requests[1]["messages"][2]["content"]
    assert result["is_error"] is True and "cut off" in result["content"]
    assert not (ws.root / "big.txt").exists()


def test_refusal_is_reported(ws):
    assert "[stopped: refusal]" in Agent(FakeClient(reply("refusal")), ws).send("go")


def test_trace_shows_requests_and_tool_calls(ws):
    lines = []
    client = FakeClient(reply("tool_use", tool("t1", "run_bash", command="echo traced")), reply("end_turn", text("ok")))
    Agent(client, ws, trace=True, log=lines.append).send("go")
    log = "\n".join(lines)
    for marker in ("REQUEST 1", "RESPONSE 1  stop_reason=tool_use", "TOOL run_bash", "REQUEST 2", "traced"):
        assert marker in log


def test_denied_call_is_not_run_and_model_is_told_why(ws):
    client = FakeClient(
        reply("tool_use", tool("t1", "write_file", path="a.py", content="x")),
        reply("end_turn", text("ok, I won't")),
    )
    perms = Permissions(ask_user=lambda tool, args: ("no", "not that file"))
    assert Agent(client, ws, permissions=perms).send("go") == "ok, I won't"
    assert not (ws.root / "a.py").exists()
    [result] = client.requests[1]["messages"][2]["content"]
    assert result["is_error"] is True and "not that file" in result["content"]


def test_approved_call_runs(ws):
    client = FakeClient(reply("tool_use", tool("t1", "write_file", path="a.py", content="x")),
                        reply("end_turn", text("done")))
    perms = Permissions(ask_user=lambda tool, args: ("yes", ""))
    Agent(client, ws, permissions=perms).send("go")
    assert (ws.root / "a.py").read_text() == "x"


def test_without_a_human_risky_calls_are_denied(ws):
    client = FakeClient(reply("tool_use", tool("t1", "run_bash", command="touch x")), reply("end_turn", text("ok")))
    agent_obj = agent_mod.Agent(client, ws)   # default permissions, no ask_user
    agent_obj.send("go")
    assert not (ws.root / "x").exists()


def test_trace_shows_permission_decisions(ws):
    lines = []
    client = FakeClient(reply("tool_use", tool("t1", "run_bash", command="git status")), reply("end_turn", text("ok")))
    agent_mod.Agent(client, ws, trace=True, log=lines.append).send("go")
    assert any("PERMISSION run_bash: allow" in line for line in lines)
