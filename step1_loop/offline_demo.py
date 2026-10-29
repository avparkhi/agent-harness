"""Watch a full traced run with no API key.

Starts a tiny local server that pretends to be the Messages API and plays back three
scripted responses. The real anthropic SDK talks to it over HTTP, so the trace shows
exactly the JSON a live run would send and receive.

Run:  python offline_demo.py
"""
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic

from agent import Agent, Workspace

SCRIPT = [
    {"stop_reason": "tool_use", "content": [
        {"type": "thinking", "thinking": "I'll write the file, then run it to check.", "signature": "demo"},
        {"type": "tool_use", "id": "toolu_1", "name": "write_file",
         "input": {"path": "hello.py", "content": "print('hi')\n"}}]},
    {"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "toolu_2", "name": "run_bash", "input": {"command": "python3 hello.py"}}]},
    {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Created hello.py. Running it prints hi."}]},
]


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


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="not-needed", base_url=f"http://127.0.0.1:{server.server_port}")
    with tempfile.TemporaryDirectory() as tmp:
        answer = Agent(client, Workspace(tmp), trace=True).send("Create hello.py that prints hi, and run it.")
        print(f"\nFINAL ANSWER: {answer}")
