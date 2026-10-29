"""Watch planning and parallel subagents, with no API key.

A local fake API plays three scripted agents: the main agent, and two explore subagents
it starts in parallel. Each agent's script is picked by the first message of its request,
just as each real subagent starts its own conversation with the brief it was given.

Run:  python offline_demo.py           (add --trace to see every request)
"""
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

from agent import Agent, Workspace
from permissions import Permissions
from sandbox import pick_sandbox

USER = "Audit this project: list every TODO comment and every place a secret might be hardcoded."
BRIEF_TODO = ("Search every .py file in this project for TODO comments. Report each one as "
              "path:line and the comment text. Read-only; don't change anything.")
BRIEF_SECRETS = ("Look for hardcoded secrets in this project's .py files: API keys, passwords, tokens "
                 "assigned as string literals. Report path:line and what it looks like, and say how "
                 "confident you are for each.")


def call(*calls):
    return {"stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]}


def say(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


PLAN = [{"content": "Find TODO comments", "status": "in_progress"},
        {"content": "Find hardcoded secrets", "status": "pending"},
        {"content": "Write the audit summary", "status": "pending"}]

SCRIPTS = {
    "main": [
        call(("m1", "todo_write", {"todos": PLAN})),
        call(("m2", "task", {"description": "find TODOs", "prompt": BRIEF_TODO, "agent_type": "explore"}),
             ("m3", "task", {"description": "find secrets", "prompt": BRIEF_SECRETS, "agent_type": "explore"})),
        call(("m4", "todo_write", {"todos": [{"content": t["content"], "status": "completed"} for t in PLAN[:2]]
                                   + [{"content": "Write the audit summary", "status": "in_progress"}]})),
        say("Audit done. 2 TODOs: app/db.py:3 (connection pooling) and app/api.py:7 (rate limiting). "
            "1 likely secret: app/settings.py:2 has an API key as a string literal; move it to an "
            "environment variable."),
    ],
    BRIEF_TODO: [
        call(("a1", "search_files", {"pattern": "TODO", "path": ".", "glob": "*.py"})),
        say("2 TODOs:\n- app/db.py:3: TODO: add connection pooling\n- app/api.py:7: TODO: rate limiting"),
    ],
    BRIEF_SECRETS: [
        call(("b1", "search_files", {"pattern": r"(?i)(key|token|password|secret)\s*=\s*['\"]", "path": ".", "glob": "*.py"})),
        call(("b2", "read_file", {"path": "app/settings.py"})),
        say("1 likely secret: app/settings.py:2 assigns API_KEY a literal that looks like a live key "
            "(high confidence). DEBUG_PASSWORD on line 3 is empty (not a secret)."),
    ],
}
SIZES = {key: [] for key in SCRIPTS}
LOCK = threading.Lock()


class FakeAPI(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["content-length"]))
        req = json.loads(raw)
        first = req["messages"][0]["content"]
        key = first if first in SCRIPTS else "main"
        with LOCK:
            SIZES[key].append(len(raw) // 4)
            step = SCRIPTS[key].pop(0)
        body = json.dumps({"id": "msg", "type": "message", "role": "assistant", "model": req["model"],
                           "stop_sequence": None, "usage": {"input_tokens": len(raw) // 4, "output_tokens": 50},
                           **step}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def make_project(root: Path) -> None:
    (root / "app").mkdir()
    (root / "app/db.py").write_text("import sqlite3\n\n# TODO: add connection pooling\ndef connect():\n"
                                    "    return sqlite3.connect('app.db')\n")
    (root / "app/api.py").write_text("from app.db import connect\n\n\ndef handler(request):\n"
                                     "    db = connect()\n    rows = db.execute('select 1').fetchall()\n"
                                     "    # TODO: rate limiting\n    return rows\n")
    (root / "app/settings.py").write_text("DEBUG = False\nAPI_KEY = 'sk-live-51Hx9demo0nly00000'\nDEBUG_PASSWORD = ''\n")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="not-needed", base_url=f"http://127.0.0.1:{server.server_port}")
    with tempfile.TemporaryDirectory() as tmp:
        make_project(Path(tmp))
        ws = Workspace(tmp, pick_sandbox())
        agent = Agent(client, ws, Permissions.load(ws.root), trace="--trace" in sys.argv)
        print(f"you> {USER}")
        print(f"\nagent> {agent.send(USER)}")
        print(f"\n[{agent.meter.summary()}]  (this demo doesn't simulate the cache)")
        print("\nRequest sizes in tokens (approximate), per agent:")
        for key, sizes in SIZES.items():
            label = "main" if key == "main" else ("todo subagent" if key == BRIEF_TODO else "secrets subagent")
            print(f"  {label:17} {sizes}")
        print("The subagents' search results and file reads stayed in their own contexts;\n"
              "the main agent only ever received their two short reports.")
