"""Durability tests. Each crash is a BaseException raised at a chosen point, which escapes
the harness the way a kill would; then a *new* Agent opens the same file and resumes."""
import copy
import socket
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock, Usage

import durable
from agent import Agent, Workspace, to_json
from durable import LeaseHeld, Parked, Session, Store, fold
from permissions import Permissions
from sandbox import NoSandbox


class Crash(BaseException):
    """Not an Exception, so nothing in the harness catches it: like the process dying."""


def text(t):
    return TextBlock(type="text", text=t)


def tool(id, name, **input):
    return ToolUseBlock(type="tool_use", id=id, name=name, input=input)


def reply(stop_reason, *content, tokens=10):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=Usage(input_tokens=tokens, output_tokens=5))


class Client:
    def __init__(self, *responses):
        self.responses, self.requests, self.messages = list(responses), [], self

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


DEAD = f"{socket.gethostname()}:999999"   # a process on this machine that doesn't exist


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    return Workspace(tmp_path, NoSandbox())


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "sessions.db")


def agent(client, ws, session, **kw):
    perms = Permissions(mode="bypass")
    return Agent(client, ws, perms, session=session, log=lambda *_: None, **kw)


def first_run(store, ws, client, **kw):
    """Start a session as a process that then dies, leaving the lease to a dead pid."""
    session = Session.create(store, owner=DEAD)
    a = agent(client, ws, session, **kw)
    with pytest.raises(Crash):
        a.send("go", request_id="r1")
    return session.id


def reopen(store, ws, sid, client, **kw):
    return agent(client, ws, Session.open(store, sid), **kw)


def kinds(store, sid):
    return [k for k, _ in store.entries(sid)]


def test_a_clean_run_logs_every_step_in_order(store, ws):
    session = Session.create(store)
    a = agent(Client(reply("tool_use", tool("t1", "read_file", path="a.txt")), reply("end_turn", text("It says alpha."))),
              ws, session)
    assert a.send("read a.txt") == "It says alpha."
    assert kinds(store, session.id) == ["user", "assistant", "intent", "result", "assistant", "final"]
    assert fold(store.entries(session.id)).messages == to_json(a.messages)


def test_crash_during_the_model_call_loses_nothing(store, ws):
    sid = first_run(store, ws, Client(reply("tool_use", tool("t1", "read_file", path="a.txt")), Crash()))
    client = Client(reply("end_turn", text("It says alpha.")))
    assert reopen(store, ws, sid, client).resume() == "It says alpha."
    # The resent request carries the finished read: nothing was redone.
    sent = client.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[2]["content"][0]["content"] == "alpha"
    assert kinds(store, sid).count("intent") == 1


def test_a_shell_command_cut_off_mid_run_is_reported_not_rerun(store, ws, monkeypatch):
    ran = []
    real = Workspace.run

    def run(self, name, args):
        ran.append(name)
        if name == "run_bash" and len(ran) == 1:
            raise Crash()
        return real(self, name, args)

    monkeypatch.setattr(Workspace, "run", run)
    sid = first_run(store, ws, Client(reply("tool_use", tool("t1", "run_bash", command="git commit -m x"))))
    client = Client(reply("end_turn", text("I'll check git log first.")))
    reopen(store, ws, sid, client).resume()
    result = client.requests[0]["messages"][-1]["content"][0]
    assert result["is_error"] and result["content"].startswith("Interrupted")
    assert ran == ["run_bash"]                         # never run a second time


def test_a_replay_safe_call_cut_off_mid_run_simply_reruns(store, ws, monkeypatch):
    real, calls = Workspace.run, []

    def run(self, name, args):
        calls.append(name)
        if len(calls) == 1:
            raise Crash()
        return real(self, name, args)

    monkeypatch.setattr(Workspace, "run", run)
    sid = first_run(store, ws, Client(reply("tool_use", tool("t1", "read_file", path="a.txt"))))
    client = Client(reply("end_turn", text("ok")))
    reopen(store, ws, sid, client).resume()
    assert calls == ["read_file", "read_file"]
    assert client.requests[0]["messages"][-1]["content"][0]["content"] == "alpha"


def test_a_round_resumes_call_by_call(store, ws, monkeypatch):
    """Three calls: the first finished, the second was mid-run, the third never started."""
    real, seen = Workspace.run, []

    def run(self, name, args):
        seen.append(args.get("command") or args.get("path"))
        if args.get("command") == "make deploy" and seen.count("make deploy") == 1:
            raise Crash()
        return real(self, name, args)

    monkeypatch.setattr(Workspace, "run", run)
    sid = first_run(store, ws, Client(reply("tool_use", tool("t1", "read_file", path="a.txt"),
                                            tool("t2", "run_bash", command="make deploy"),
                                            tool("t3", "write_file", path="b.txt", content="beta"))))
    client = Client(reply("end_turn", text("ok")))
    reopen(store, ws, sid, client).resume()
    assert seen == ["a.txt", "make deploy", "b.txt"]   # only the call that never started ran now
    results = client.requests[0]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
    assert results[0]["content"] == "alpha" and results[1]["content"].startswith("Interrupted")
    assert (ws.root / "b.txt").read_text() == "beta"


def test_thinking_blocks_survive_a_restart_byte_for_byte(store, ws):
    thought = ThinkingBlock(type="thinking", thinking="read it first", signature="sig-abc123")
    sid = first_run(store, ws, Client(reply("tool_use", thought, tool("t1", "read_file", path="a.txt")), Crash()))
    client = Client(reply("end_turn", text("ok")))
    reopen(store, ws, sid, client).resume()
    assert client.requests[0]["messages"][1]["content"][0] == \
        {"type": "thinking", "thinking": "read it first", "signature": "sig-abc123"}


def test_the_system_prompt_is_frozen_with_the_session(store, ws):
    sid = first_run(store, ws, Client(reply("tool_use", tool("t1", "read_file", path="a.txt")), Crash()))
    (ws.root / "HARNESS.md").write_text("- a note added after the session started\n")
    client = Client(reply("end_turn", text("ok")))
    reopen(store, ws, sid, client).resume()
    assert "added after" not in client.requests[0]["system"][0]["text"]


def test_the_same_request_id_runs_once(store, ws):
    session = Session.create(store)
    client = Client(reply("end_turn", text("first answer")))
    a = agent(client, ws, session)
    assert a.send("hello", request_id="r1") == "first answer"
    assert a.send("hello", request_id="r1") == "first answer"      # a retry: no second model call
    assert len(client.requests) == 1 and kinds(store, session.id).count("user") == 1


def test_a_live_lease_blocks_a_second_process(store, ws):
    session = Session.create(store, owner="other-host:42")
    with pytest.raises(LeaseHeld):
        Session.open(store, session.id, owner="me:1")


def test_an_expired_lease_or_a_dead_local_process_can_be_taken_over(store, monkeypatch):
    s = Session.create(store, owner="other-host:42")
    Session(store, s.id, owner="me:1").acquire(now=durable.time.time() + durable.LEASE_SECONDS + 1)
    t = Session.create(store, owner=DEAD)
    Session.open(store, t.id, owner="me:1")                         # no wait: that pid is gone


def test_a_process_that_lost_its_lease_cannot_write(store):
    old = Session.create(store, owner="me:1")
    Session(store, old.id, owner="other:2").acquire(now=durable.time.time() + 999)
    with pytest.raises(LeaseHeld):
        old.append(("user", {"content": "late write"}))
    assert store.entries(old.id) == []


def test_a_session_that_keeps_crashing_is_parked(store, ws):
    sid = first_run(store, ws, Client(Crash()))
    for _ in range(durable.RESTART_LIMIT):
        with pytest.raises(Crash):
            reopen(store, ws, sid, Client(Crash())).resume()
    with pytest.raises(Parked):
        reopen(store, ws, sid, Client()).resume()
    assert reopen(store, ws, sid, Client(reply("end_turn", text("ok")))).resume(force=True) == "ok"


def test_a_failed_commit_writes_nothing(store):
    session = Session.create(store)

    def half(db):
        db.execute("insert into entries values (?, 1, 'user', '{}', 0)", (session.id,))
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        store.tx(half)
    assert store.entries(session.id) == []


def test_plan_cost_and_compaction_come_back_after_a_restart(store, ws):
    todos = [{"content": "read a.txt", "status": "in_progress"}]
    summary = reply("end_turn", text("<summary>User wants a.txt read.</summary>"), tokens=10)
    sid = first_run(store, ws, Client(
        reply("tool_use", tool("t1", "todo_write", todos=todos), tokens=5000),
        summary,                                   # compaction runs at the next safe point
        Crash()), compact_at=1000)
    a = reopen(store, ws, sid, Client(reply("end_turn", text("ok"))), compact_at=1000)
    assert a.todos == todos
    assert a.meter.requests == 2
    assert len(a.messages) == 1 and "User wants a.txt read." in a.messages[0]["content"][0]["text"]
    assert a.resume() == "ok"


def test_list_shows_where_each_session_stands(store, ws):
    done = Session.create(store)
    agent(Client(reply("end_turn", text("hi"))), ws, done).send("hello")
    first_run(store, ws, Client(reply("tool_use", tool("t1", "read_file", path="a.txt")), Crash()))
    rows = store.list()
    assert [(r["status"], r["first_prompt"]) for r in rows] == [("idle", "hello"), ("needs_model", "go")]
