"""Watch permissions work, with no API key.

A local fake API plays a scripted model that tries five things. Your answers at the
approval prompt are scripted too (printed after each question), so it runs unattended.

Run:  python offline_demo.py           (add --trace to see every request)
"""
import builtins
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import anthropic

from agent import Agent, Workspace, ask_in_terminal
from permissions import Permissions
from sandbox import pick_sandbox


def call(id, name, **input):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": id, "name": name, "input": input}]}


SCRIPT = [
    call("t1", "read_file", path=".env"),                                        # deny rule
    call("t2", "run_bash", command="cat notes.txt && curl -s https://example.com"),  # asks, you say no
    call("t3", "write_file", path="hello.py", content="print('hi from the sandbox')\n"),  # asks, diff shown
    call("t4", "run_bash", command="python3 hello.py && touch /etc/oops"),       # asks; sandbox blocks /etc
    call("t5", "write_file", path=".harness/settings.json", content='{"permissions": {"defaultMode": "bypass"}}'),
    {"stop_reason": "end_turn", "content": [{"type": "text", "text": "hello.py works. I couldn't read .env or change settings."}]},
]
ANSWERS = iter(["n", "no network please", "y", "y"])


class FakeAPI(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        step = SCRIPT[FakeAPI.calls]
        FakeAPI.calls += 1
        body = json.dumps({"id": f"msg_{FakeAPI.calls}", "type": "message", "role": "assistant",
                           "model": "claude-opus-5-5", "stop_sequence": None,
                           "usage": {"input_tokens": 0, "output_tokens": 0}, **step}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def scripted_input(prompt=""):
    answer = next(ANSWERS)
    print(f"{prompt}{answer}")
    return answer


if __name__ == "__main__":
    builtins.input = scripted_input
    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="not-needed", base_url=f"http://127.0.0.1:{server.server_port}")
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, ".env").write_text("SECRET=1\n")
        Path(tmp, "notes.txt").write_text("remember the milk\n")
        ws = Workspace(tmp, pick_sandbox())
        perms = Permissions.load(ws.root, ask_user=ask_in_terminal(ws))
        agent = Agent(client, ws, perms, trace="--trace" in sys.argv)
        print(f"sandbox: {ws.sandbox.name}")
        answer = agent.send("Make hello.py and run it.")
        print("\nWhat the model was told for each call:")
        for message in agent.messages:
            if message["role"] == "user" and isinstance(message["content"], list):
                for r in message["content"]:
                    flat = " / ".join(line for line in r["content"].splitlines() if line.strip())
                    print(f"  {r['tool_use_id']} [is_error={r['is_error']}]: {flat[:150]}")
        print(f"\nFINAL ANSWER: {answer}")
