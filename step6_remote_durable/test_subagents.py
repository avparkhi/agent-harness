"""Tests for planning (todo_write), search_files and subagents (task)."""
import copy
import threading
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock, Usage

import context
import subagents
from agent import Agent, Workspace
from permissions import Permissions
from sandbox import NoSandbox


def text(t):
    return TextBlock(type="text", text=t)


def tool(id, name, **input):
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input)


def reply(stop_reason, *content, tokens=100):
    usage = Usage(input_tokens=tokens, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage)


class RoutedClient:
    """Each agent (main or subagent) gets its own script, chosen by the first user message
    of the request: "main" for the user's message, otherwise the subagent's brief. Thread
    safe, because parallel subagents call it from different threads."""

    def __init__(self, scripts, main_prompt):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.main_prompt = main_prompt
        self.requests = {k: [] for k in scripts}
        self.lock = threading.Lock()
        self.messages = self

    def create(self, **kwargs):
        first = kwargs["messages"][0]["content"]
        key = first if isinstance(first, str) and first != self.main_prompt and first in self.scripts else "main"
        with self.lock:
            self.requests[key].append(copy.deepcopy(kwargs))
            step = self.scripts[key].pop(0)
        return step() if callable(step) else step


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "USER_MEMORY", tmp_path / "none.md")
    return Workspace(tmp_path, NoSandbox())


# --- planning -------------------------------------------------------------------------------

def test_todo_list_is_saved_shown_and_echoed(ws):
    todos = [{"content": "write parser", "status": "in_progress"}, {"content": "add tests", "status": "pending"}]
    logs = []
    client = RoutedClient({"main": [reply("tool_use", tool("t1", "todo_write", todos=todos)),
                                    reply("end_turn", text("ok"))]}, "plan it")
    agent = Agent(client, ws, log=logs.append)
    agent.send("plan it")
    assert agent.todos == todos
    assert any("[>] write parser\n[ ] add tests" in line for line in logs)
    [result] = client.requests["main"][1]["messages"][2]["content"]
    assert result["is_error"] is False and "[ ] add tests" in result["content"]


def test_todo_list_allows_one_item_in_progress(ws):
    agent = Agent(RoutedClient({}, ""), ws)
    out, is_error = agent._todo_write([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}])
    assert is_error and "only one" in out and agent.todos == []


def test_todo_list_survives_compaction(ws):
    todos = [{"content": "step one", "status": "completed"}, {"content": "step two", "status": "in_progress"}]
    client = RoutedClient({"main": [
        reply("tool_use", tool("t1", "todo_write", todos=todos), tokens=5000),
        reply("end_turn", text("<summary>did step one</summary>")),
        reply("end_turn", text("done")),
    ]}, "go")
    agent = Agent(client, ws, compact_at=1000, log=lambda *_: None)
    agent.send("go")
    [only] = client.requests["main"][2]["messages"]
    assert "[x] step one\n[>] step two" in only["content"][0]["text"]


def test_planning_tools_never_ask(ws):
    p = Permissions.load(ws.root)
    assert p.check("todo_write", {"todos": []}).action == "allow"
    assert p.check("search_files", {"pattern": "x", "path": ".", "glob": "*"}).action == "allow"
    assert p.check("task", {"description": "d", "prompt": "p", "agent_type": "general"}).action == "allow"


# --- search_files ---------------------------------------------------------------------------

def test_search_files(ws):
    (ws.root / "src").mkdir()
    (ws.root / "src/app.py").write_text("import os\ndef main():\n    return 42\n")
    (ws.root / "src/notes.txt").write_text("def main is documented here\n")
    (ws.root / ".git").mkdir()
    (ws.root / ".git/config").write_text("def main\n")
    assert ws.search_files(r"def main", ".", "*.py") == "src/app.py:2: def main():"
    assert "notes.txt" in ws.search_files(r"def main", "src", "*")
    assert ".git" not in ws.search_files(r"def main", ".", "*")
    assert ws.search_files("nothing-matches-this", ".", "*") == "no matches"
    assert ws.run("search_files", {"pattern": "x", "path": "..", "glob": "*"})[1] is True


def test_search_files_limit(ws):
    (ws.root / "big.txt").write_text("hit\n" * 500)
    out = ws.search_files("hit", ".", "*", limit=10)
    assert out.count("big.txt:") == 10 and "narrow the search" in out


# --- subagents ------------------------------------------------------------------------------

BRIEF = "Find where the config is loaded. Report file and line."


def test_explore_subagent_has_fresh_context_and_fewer_tools(ws):
    (ws.root / "config.py").write_text("def load_config():\n    return {}\n")
    client = RoutedClient({
        "main": [reply("tool_use", text("I'll delegate."), tool("t1", "task", description="find config",
                                                                prompt=BRIEF, agent_type="explore")),
                 reply("end_turn", text("Config loads in config.py:1."))],
        BRIEF: [reply("tool_use", tool("s1", "search_files", pattern="def load_config", path=".", glob="*.py")),
                reply("tool_use", tool("s2", "read_file", path="config.py")),
                reply("end_turn", text("config.py:1 defines load_config()."))],
    }, "where is config loaded?")
    agent = Agent(client, ws, log=lambda *_: None)
    assert agent.send("where is config loaded?") == "Config loads in config.py:1."

    child_first = client.requests[BRIEF][0]
    assert child_first["messages"] == [{"role": "user", "content": BRIEF}]        # nothing from the parent
    assert [t["name"] for t in child_first["tools"]] == ["read_file", "search_files"]
    assert "read-only research subagent" in child_first["system"][0]["text"]
    # The parent gets the report only; the file contents the child read never enter its context.
    [result] = client.requests["main"][1]["messages"][2]["content"]
    assert result["content"] == "config.py:1 defines load_config()." and result["is_error"] is False
    assert "return {}" not in str(client.requests["main"][1]["messages"])


def test_explore_subagent_cannot_write_even_if_it_tries(ws):
    client = RoutedClient({
        "main": [reply("tool_use", tool("t1", "task", description="x", prompt=BRIEF, agent_type="explore")),
                 reply("end_turn", text("ok"))],
        BRIEF: [reply("tool_use", tool("s1", "write_file", path="hack.py", content="x")),
                reply("end_turn", text("couldn't write"))],
    }, "go")
    Agent(client, ws, permissions=Permissions("bypass"), log=lambda *_: None).send("go")
    assert not (ws.root / "hack.py").exists()
    [denied] = client.requests[BRIEF][1]["messages"][2]["content"]
    assert "read-only" in denied["content"]


def test_general_subagent_shares_permissions_and_cannot_spawn_more(ws):
    client = RoutedClient({
        "main": [reply("tool_use", tool("t1", "task", description="x", prompt=BRIEF, agent_type="general")),
                 reply("end_turn", text("ok"))],
        BRIEF: [reply("tool_use", tool("s1", "write_file", path="new.py", content="x = 1\n")),
                reply("end_turn", text("wrote new.py"))],
    }, "go")
    approvals = []
    perms = Permissions(ask_user=lambda tool, args: approvals.append(tool) or ("yes", ""))
    Agent(client, ws, permissions=perms, log=lambda *_: None).send("go")
    assert approvals == ["write_file"]                       # the subagent's write still asked the user
    assert (ws.root / "new.py").exists()
    assert "task" not in [t["name"] for t in client.requests[BRIEF][0]["tools"]]


def test_parallel_subagents_run_at_the_same_time(ws):
    both_running = threading.Barrier(2, timeout=5)

    def wait_then(report):
        def step():
            both_running.wait()      # raises if the other subagent isn't running concurrently
            return reply("end_turn", text(report))
        return step

    client = RoutedClient({
        "main": [reply("tool_use",
                       tool("t1", "task", description="a", prompt="job A", agent_type="explore"),
                       tool("t2", "task", description="b", prompt="job B", agent_type="explore")),
                 reply("end_turn", text("both done"))],
        "job A": [wait_then("report A")],
        "job B": [wait_then("report B")],
    }, "go")
    agent = Agent(client, ws, log=lambda *_: None)
    assert agent.send("go") == "both done"
    results = client.requests["main"][1]["messages"][2]["content"]
    assert [r["content"] for r in results] == ["report A", "report B"]    # order kept
    assert agent.meter.requests == 4 and agent.meter.subagent_cost > 0


def test_subagent_that_hits_its_turn_cap_is_an_error(ws, monkeypatch):
    monkeypatch.setattr(subagents, "SUBAGENT_MAX_TURNS", 2)
    loop = [reply("tool_use", tool(f"s{i}", "search_files", pattern="x", path=".", glob="*")) for i in range(2)]
    client = RoutedClient({
        "main": [reply("tool_use", tool("t1", "task", description="x", prompt=BRIEF, agent_type="explore")),
                 reply("end_turn", text("it got stuck"))],
        BRIEF: loop,
    }, "go")
    Agent(client, ws, log=lambda *_: None).send("go")
    [result] = client.requests["main"][1]["messages"][2]["content"]
    assert result["is_error"] is True and "turn cap" in result["content"]


def test_subagent_model_override(ws):
    client = RoutedClient({
        "main": [reply("tool_use", tool("t1", "task", description="x", prompt=BRIEF, agent_type="explore")),
                 reply("end_turn", text("ok"))],
        BRIEF: [reply("end_turn", text("report"))],
    }, "go")
    Agent(client, ws, subagent_model="claude-haiku-5-5", log=lambda *_: None).send("go")
    assert client.requests["main"][0]["model"] == "claude-opus-5-5"
    assert client.requests[BRIEF][0]["model"] == "claude-haiku-5-5"
