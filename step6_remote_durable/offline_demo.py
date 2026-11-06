"""Kill an agent with kill -9 in the middle of a shell command, then resume it in a new process.

No API key and no Daytona account: a local fake API plays the model, and FakeDaytona keeps
the "remote" sandbox in a folder. Everything else is the real harness, the real SDK and a
real SIGKILL.

    1. A child process starts a durable session on a (fake) remote sandbox. The model plans,
       writes two files, then runs a slow test command.
    2. While that command runs, this process kills the child with SIGKILL. No cleanup runs.
    3. We print the session log: the command's intent is there, its result isn't.
    4. A new Agent in this process resumes. It reconnects to the same sandbox (the files are
       still there), tells the model the command was interrupted, and finishes the task.

Run:  python offline_demo.py           (add --trace to see every request)
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

from agent import Agent, show_session
from durable import Session, Store
from fake_daytona import FakeDaytona
from permissions import Permissions
from remote import DaytonaWorkspace, open_sandbox

TASK = "Add a greet(name) function in greet.py with a test, and run the test."
PLAN = [{"content": "Write greet.py and its test", "status": "in_progress"},
        {"content": "Run the test", "status": "pending"}]
SEEN = {"thinking_resent": False}


def call(*calls, thinking=None):
    content = [{"type": "thinking", "thinking": thinking, "signature": "demo-signature"}] if thinking else []
    return {"stop_reason": "tool_use",
            "content": content + [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]}


def model(messages):
    """The fake model's whole script, picked by how far the conversation has got."""
    turn = sum(m["role"] == "assistant" for m in messages)
    if turn == 0:
        return call(("t1", "todo_write", {"todos": PLAN}),
                    ("t2", "write_file", {"path": "greet.py", "content": "def greet(name):\n    return f'Hello, {name}!'\n"}),
                    ("t3", "write_file", {"path": "test_greet.py", "content":
                        "from greet import greet\n\nassert greet('Ada') == 'Hello, Ada!'\nprint('test passed')\n"}),
                    thinking="Plan first, then write the code and its test.")
    if turn == 1:
        return call(("t4", "run_bash", {"command": "sleep 5; python3 test_greet.py"}))
    if turn == 2:
        SEEN["thinking_resent"] = any(b.get("signature") == "demo-signature" for b in messages[1]["content"])
        last = messages[-1]["content"][0]
        if last["is_error"] and "Interrupted" in last["content"]:
            return {"stop_reason": "tool_use", "content": [
                {"type": "text", "text": "The test run was cut off. Tests are safe to rerun, so I'll run them again."},
                {"type": "tool_use", "id": "t5", "name": "run_bash", "input": {"command": "python3 test_greet.py"}}]}
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Added greet.py and test_greet.py; the test passes."}]}


class FakeAPI(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        body = json.dumps({"id": "msg", "type": "message", "role": "assistant", "model": req["model"],
                           "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20},
                           **model(req["messages"])}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def build(port, db, cloud, sandbox_id=None, session_id=None, log=print):
    client = anthropic.Anthropic(api_key="not-needed", base_url=f"http://127.0.0.1:{port}")
    store = Store(db)
    session = Session.open(store, session_id) if session_id else Session.create(store, {"cwd": "demo"})
    sandbox = open_sandbox(FakeDaytona(cloud), sandbox_id or session.meta.get("sandbox_id"),
                           session_id=session.id, log=log)
    session.set_meta(sandbox_id=sandbox.id)
    permissions = Permissions(mode="accept-edits", allow=["run_bash(sleep *)", "run_bash(python3 *)"])
    return Agent(client, DaytonaWorkspace(sandbox), permissions, session=session, log=log,
                 trace="--trace" in sys.argv), store


def child(port, db, cloud):
    agent, _ = build(int(port), db, cloud)
    print(f"session {agent.session.id} on {agent.workspace.label}", flush=True)
    agent.send(TASK)


def wait_for_intent(db, tool, timeout=20):
    store, end = Store(db), time.time() + timeout
    while time.time() < end:
        for (sid,) in store.db.execute("select id from sessions"):
            if any(k == "intent" and d["name"] == tool for k, d in store.entries(sid)):
                return sid
        time.sleep(0.1)
    raise SystemExit(f"the child never started {tool}")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        db, cloud = f"{tmp}/sessions.db", f"{tmp}/cloud"
        print(f"you> {TASK}\n")
        print("--- 1. a child process starts the session")
        proc = subprocess.Popen([sys.executable, __file__, "--child", str(server.server_port), db, cloud],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sid = wait_for_intent(db, "run_bash")
        time.sleep(0.5)          # let the command get going
        print("--- 2. the shell command is running: kill -9 the child")
        os.kill(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        print("\n".join(f"   child| {line}" for line in out.splitlines()))
        print(f"   child exit code: {proc.returncode} (killed; nothing got to clean up)\n")

        print("--- 3. the session log, read from disk")
        store = Store(db)
        show_session(store, sid)
        print("   -> run_bash has an intent and no result: it may or may not have finished.\n")

        print("--- 4. a new process resumes")
        agent, store = build(server.server_port, db, cloud, session_id=sid)
        print(f"   files already in the sandbox: {agent.workspace.run('run_bash', {'command': 'ls'})[0].split('output:')[1].split()}")
        answer = agent.resume()
        print(f"\nagent> {answer}")
        print(f"\n[the request after the restart resent the pre-crash thinking block unchanged: "
              f"{'yes' if SEEN['thinking_resent'] else 'NO'}]")
        print(f"[{agent.meter.summary()}]  (the meter was restored from the log, so it counts both processes)")

        print("\n--- 5. the whole log")
        show_session(store, sid)
        agent.session.close()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:5])
    else:
        main()
