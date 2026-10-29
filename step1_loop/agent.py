"""mini-harness, step 1: the bare agent loop.

Every coding agent (Claude Code, Codex, ...) is built on this loop:
    1. send the conversation (system prompt + tool schemas + messages) to the model
    2. if the model asked for tools, run them and append the results as a user turn
    3. repeat until the model answers without asking for a tool

Run:  python agent.py --cwd ./scratch --trace
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
EFFORT = "high"          # Opus 5.5 defaults to "medium"; effort is how hard it thinks
MAX_TURNS = 25           # one turn = one model call; a cap stops runaway loops
BASH_TIMEOUT = 30        # seconds
MAX_RESULT_CHARS = 20_000  # long tool output is cut so it can't flood the context

SYSTEM_PROMPT = """You are a coding agent working in the directory {cwd}.
Use the tools to read files, write files and run shell commands there.
Read a file before you change it. After changing code, run it or its tests to check your work.
When the task is done, reply with a short summary of what you did."""

# The model never sees our Python functions, only these schemas. The description is the
# tool's documentation for the model, so it matters as much as the code behind it.
TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its full contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the working directory."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content. Parent folders are created.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory."},
                "content": {"type": "string", "description": "The complete new file content."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "run_bash",
        "description": f"Run a bash command in the working directory. Returns exit code, stdout and stderr. "
                       f"Times out after {BASH_TIMEOUT} seconds, so don't start servers or interactive programs.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class Workspace:
    """Runs tool calls. Guardrails here are deliberately minimal: file tools are confined to
    the working directory and bash has a timeout. bash itself can still touch anything the
    process can (try `cat /etc/passwd`). Real permissions and sandboxing are step two."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError(f"{path} is outside the working directory")
        return p

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text()

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def run_bash(self, command: str) -> str:
        r = subprocess.run(["bash", "-c", command], cwd=self.root, capture_output=True,
                           text=True, timeout=BASH_TIMEOUT)
        return f"exit code: {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (output, is_error). Errors go back to the model as text, not exceptions:
        a model that sees 'file not found' can recover; a crashed harness can't."""
        handler = {"read_file": self.read_file, "write_file": self.write_file, "run_bash": self.run_bash}.get(name)
        if handler is None:
            return f"unknown tool: {name}", True
        try:
            out, is_error = handler(**args), False
        except subprocess.TimeoutExpired:
            out, is_error = f"command timed out after {BASH_TIMEOUT}s", True
        except Exception as e:
            out, is_error = f"{type(e).__name__}: {e}", True
        if len(out) > MAX_RESULT_CHARS:
            out = out[:MAX_RESULT_CHARS] + f"\n[... cut {len(out) - MAX_RESULT_CHARS} characters]"
        return out, is_error


def to_json(x):
    """SDK responses are pydantic objects; turn them into plain JSON for the trace."""
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json", exclude_none=True)
    if isinstance(x, list):
        return [to_json(i) for i in x]
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    return x


class Agent:
    def __init__(self, client, workspace: Workspace, model: str = MODEL,
                 max_turns: int = MAX_TURNS, trace: bool = False, log=print):
        self.client, self.workspace, self.model = client, workspace, model
        self.max_turns, self.trace, self.log = max_turns, trace, log
        self.system = SYSTEM_PROMPT.format(cwd=workspace.root)
        self.messages: list[dict] = []   # the whole conversation; the API itself is stateless

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        for turn in range(1, self.max_turns + 1):
            if self.trace:
                self._show(f"REQUEST {turn}", {"system": self.system if turn == 1 else "(unchanged)",
                                               "tools": [t["name"] for t in TOOLS], "messages": self.messages})
            response = self.client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=self.system, tools=TOOLS,
                messages=self.messages, output_config={"effort": EFFORT},
                thinking={"type": "adaptive", "display": "summarized"},  # so the trace shows reasoning
            )
            # Append the assistant turn exactly as it came back, thinking blocks included.
            self.messages.append({"role": "assistant", "content": response.content})
            if self.trace:
                self._show(f"RESPONSE {turn}  stop_reason={response.stop_reason}",
                           {"content": response.content, "usage": getattr(response, "usage", None)})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "tool_use":
                results = [self._run_tool(call) for call in tool_calls]
            elif tool_calls:
                # e.g. max_tokens cut the turn off mid-call. Every tool_use still needs a
                # tool_result, or the next request is rejected, so answer them with errors.
                results = [{"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                            "content": f"not run: your turn was cut off (stop_reason={response.stop_reason})"}
                           for c in tool_calls]
            else:
                # end_turn (done), or max_tokens/refusal with no pending tool call: hand back to the user.
                text = "".join(b.text for b in response.content if b.type == "text")
                return text if response.stop_reason == "end_turn" else f"{text}\n[stopped: {response.stop_reason}]"
            # Tool results go back as a *user* turn: only the user side can carry tool_result blocks.
            self.messages.append({"role": "user", "content": results})
        return f"[stopped: hit the {self.max_turns}-turn cap]"

    def _run_tool(self, call) -> dict:
        out, is_error = self.workspace.run(call.name, call.input)
        if self.trace:
            self._show(f"TOOL {call.name}{'  (error)' if is_error else ''}", {"input": call.input, "output": out})
        return {"type": "tool_result", "tool_use_id": call.id, "content": out, "is_error": is_error}

    def _show(self, title: str, payload) -> None:
        self.log(f"\n===== {title} =====\n{json.dumps(to_json(payload), indent=2)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="A minimal coding agent: one loop, three tools.")
    ap.add_argument("prompt", nargs="*", help="run one task and exit; omit for a REPL")
    ap.add_argument("--cwd", default=".", help="working directory the agent may edit")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--trace", action="store_true", help="print every request, response and tool call")
    args = ap.parse_args(argv)

    import anthropic  # imported here so the tests run without the SDK or an API key
    agent = Agent(anthropic.Anthropic(), Workspace(args.cwd), args.model, args.max_turns, args.trace)
    if args.prompt:
        print(agent.send(" ".join(args.prompt)))
        return
    print(f"agent in {agent.workspace.root} ({args.model}). Type 'exit' to quit.")
    while (line := input("\nyou> ").strip()) not in {"exit", "quit"}:
        if line:
            print(f"\nagent> {agent.send(line)}")


if __name__ == "__main__":
    main()
