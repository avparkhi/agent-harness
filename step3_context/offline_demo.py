"""Watch memory, caching and compaction work, with no API key.

A local fake API plays a scripted model. It also *simulates* the prompt cache: it counts
about 4 characters per token, and bills as a cache read whatever part of a request repeats
the previous request byte for byte. Real numbers differ; the shape is the same.

Run:  python offline_demo.py           (add --trace to see every request)
"""
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import anthropic

from agent import Agent, Workspace
from permissions import Permissions
from sandbox import pick_sandbox


def call(id, name, **input):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": id, "name": name, "input": input}]}


def say(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


SCRIPT = [
    # Task 1: the model follows HARNESS.md's instruction to run the tests with python3 -m pytest
    call("t1", "write_file", path="stats.py", content="def mean(xs):\n    return sum(xs) / len(xs)\n"),
    call("t2", "write_file", path="test_stats.py",
         content="from stats import mean\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n"),
    call("t3", "run_bash", command="python3 -c 'import stats; print(stats.mean([1, 2, 3]))'"),
    call("t4", "remember", note="stats.py holds the statistics helpers; each has a test in test_stats.py"),
    say("Added mean() in stats.py with a test. It prints 2.0 for [1, 2, 3]."),
    # Task 2: the context is now past --compact-at, so the harness summarizes first
    say("<summary>User asked for a stats module. Done: stats.py with mean(), test in test_stats.py, "
        "verified mean([1,2,3]) == 2.0. Saved a memory note. Next: user's new request (median).</summary>"),
    call("t5", "write_file", path="stats.py", content="def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
         "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n"),
    say("Added median() next to mean()."),
]


class FakeAPI(BaseHTTPRequestHandler):
    calls = 0
    previous = ""

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        prompt = json.dumps([req["tools"], req["system"], req["messages"]], sort_keys=True)
        shared = 0
        while shared < min(len(prompt), len(FakeAPI.previous)) and prompt[shared] == FakeAPI.previous[shared]:
            shared += 1
        FakeAPI.previous = prompt
        step = SCRIPT[FakeAPI.calls]
        FakeAPI.calls += 1
        usage = {"cache_read_input_tokens": shared // 4, "cache_creation_input_tokens": (len(prompt) - shared) // 4,
                 "input_tokens": 0, "output_tokens": len(json.dumps(step["content"])) // 4}
        body = json.dumps({"id": f"msg_{FakeAPI.calls}", "type": "message", "role": "assistant",
                           "model": "claude-opus-5-5", "stop_sequence": None, "usage": usage, **step}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="not-needed", base_url=f"http://127.0.0.1:{server.server_port}")
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "HARNESS.md").write_text("# Project memory\nRun tests with: python3 -m pytest -q\n")
        ws = Workspace(tmp, pick_sandbox())
        perms = Permissions.load(ws.root, mode="accept-edits", allow=["run_bash(python3 *)", "remember"])
        agent = Agent(client, ws, perms, trace="--trace" in sys.argv, compact_at=1000)
        print("System prompt includes HARNESS.md:", "python3 -m pytest" in agent.system[0]["text"])

        for task in ["Create stats.py with a mean() function and a test.", "Now add median()."]:
            print(f"\nyou> {task}")
            start = agent.meter.requests
            print(f"agent> {agent.send(task)}")
            print("  per request (uncached in / cache write / cache read / out):")
            for i, row in enumerate(agent.meter.history[start:], start + 1):
                print(f"    request {i}: {row}")
            print(f"  [{agent.meter.summary()}]")

        print("\nHARNESS.md now reads:\n" + Path(tmp, "HARNESS.md").read_text())
