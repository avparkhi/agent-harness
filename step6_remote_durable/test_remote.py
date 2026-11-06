"""Remote workspace tests. FakeDaytona stands in for the SDK, so no account is needed."""
import copy
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock, Usage

from agent import Agent
from fake_daytona import FakeDaytona
from permissions import Permissions
from remote import DaytonaWorkspace, download, open_sandbox, upload


class Client:
    def __init__(self, *responses):
        self.responses, self.requests, self.messages = list(responses), [], self

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def reply(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=Usage(input_tokens=10, output_tokens=5))


def tool(id, name, **input):
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input)


@pytest.fixture
def daytona(tmp_path):
    return FakeDaytona(tmp_path / "cloud")


@pytest.fixture
def ws(daytona):
    return DaytonaWorkspace(open_sandbox(daytona, session_id="s1", log=lambda *_: None))


def test_new_sandbox_is_labelled_and_auto_stops(daytona, ws):
    params = ws.sandbox.params     # the real SDK's params class when it's installed, else a dict
    get = params.get if isinstance(params, dict) else lambda k: getattr(params, k)
    assert get("labels") == {"harness-session": "s1"} and get("auto_stop_interval") == 30
    assert ws.root == "/home/daytona" and ws.sandbox.id in ws.label


def test_files_and_shell_run_in_the_sandbox(ws):
    assert ws.run("write_file", {"path": "pkg/a.py", "content": "print('hi')\n"}) == ("wrote 12 characters to pkg/a.py", False)
    assert ws.run("read_file", {"path": "pkg/a.py"}) == ("print('hi')\n", False)
    out, is_error = ws.run("run_bash", {"command": "python3 pkg/a.py; ls nope"})
    assert not is_error and "hi" in out and "No such file" in out    # stderr is shown too
    assert "exit code: 2" in out
    assert (ws.sandbox.disk / "home/daytona/pkg/a.py").exists()      # it's on the sandbox's disk


def test_search_runs_remotely_and_skips_caches(ws):
    ws.run("write_file", {"path": "a.py", "content": "x = 1  # TODO one\n"})
    ws.run("write_file", {"path": "b.txt", "content": "TODO not python\n"})
    ws.run("write_file", {"path": "__pycache__/c.py", "content": "TODO cached\n"})
    assert ws.run("search_files", {"pattern": "TODO", "path": ".", "glob": "*.py"}) == ("a.py:1: x = 1  # TODO one", False)


def test_a_hostile_search_pattern_is_just_a_pattern(ws):
    out, is_error = ws.run("search_files", {"pattern": "'\"; import os; os.system('touch pwned') #",
                                            "path": ".", "glob": "*"})
    assert (out, is_error) == ("no matches", False)
    assert not (ws.sandbox.disk / "home/daytona/pwned").exists()


def test_paths_cannot_escape_the_working_directory(ws):
    out, is_error = ws.run("write_file", {"path": "../../etc/x", "content": "no"})
    assert is_error and "outside the working directory" in out


def test_project_memory_lives_in_the_sandbox(ws):
    ws.run("remember", {"note": "run tests with pytest -q"})
    assert "- run tests with pytest -q" in ws.read_file("HARNESS.md")
    agent = Agent(Client(), ws, Permissions())
    assert "run tests with pytest -q" in agent.system[0]["text"]
    assert "Daytona sandbox" in agent.system[0]["text"]


def test_reconnect_finds_the_same_files_and_starts_a_stopped_sandbox(daytona, ws):
    ws.write_file("notes.txt", "still here")
    ws.sandbox.stop()                                   # auto-stop after idling
    again = DaytonaWorkspace(open_sandbox(daytona, ws.sandbox.id, log=lambda *_: None))
    assert again.sandbox is ws.sandbox and again.sandbox.starts == 1
    assert again.read_file("notes.txt") == "still here"


def test_upload_and_download_skip_git_and_caches(tmp_path, ws):
    src = tmp_path / "project"
    (src / "app").mkdir(parents=True)
    (src / "app/main.py").write_text("print(1)\n")
    (src / ".git").mkdir()
    (src / ".git/config").write_text("secret remote")
    assert upload(ws, src) == 1
    ws.write_file("app/new.py", "print(2)\n")
    out = tmp_path / "back"
    assert download(ws, out) == 2
    assert (out / "app/new.py").read_text() == "print(2)\n" and not (out / ".git").exists()


def test_the_agent_loop_does_not_care_where_tools_run(ws):
    client = Client(
        reply("tool_use", tool("t1", "write_file", path="hello.py", content="print('hello from the sandbox')\n")),
        reply("tool_use", tool("t2", "run_bash", command="python3 hello.py")),
        reply("end_turn", TextBlock(type="text", text="Done.")),
    )
    agent = Agent(client, ws, Permissions(mode="accept-edits", allow=["run_bash(python3 *)"]))
    assert agent.send("make hello.py and run it") == "Done."
    last = client.requests[-1]["messages"][-1]["content"][0]
    assert "hello from the sandbox" in last["content"] and not last["is_error"]
