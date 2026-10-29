"""mini-harness, step 2: the agent loop plus permissions and a sandbox.

Same loop as step 1. What's new is one checkpoint between "the model asked for a tool"
and "the tool ran" (see Agent._run_tool), and bash now runs inside an OS sandbox.
Compare:  diff ../step1_loop/agent.py agent.py

Run:  python agent.py --cwd ./scratch --trace
      python agent.py --cwd ./scratch --mode accept-edits --allow "run_bash(pytest*)"
"""
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
from pathlib import Path

from permissions import MODES, Permissions, subject_of
from sandbox import clean_env, pick_sandbox

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
EFFORT = "high"          # Opus 5.5 defaults to "medium"; effort is how hard it thinks
MAX_TURNS = 25           # one turn = one model call; a cap stops runaway loops
BASH_TIMEOUT = 30        # seconds
MAX_RESULT_CHARS = 20_000  # long tool output is cut so it can't flood the context

SYSTEM_PROMPT = """You are a coding agent working in the directory {cwd}.
Use the tools to read files, write files and run shell commands there.
Read a file before you change it. After changing code, run it or its tests to check your work.
Some tool calls need the user's approval. If one is denied, don't retry the same call:
read the reason, then change your approach or ask the user.
Shell commands run in a sandbox: no network, and writes only inside {cwd} and /tmp.
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
    """Runs tool calls. It no longer decides anything: by the time a call gets here,
    Permissions has already allowed it. Bash runs inside the sandbox."""

    def __init__(self, root: str | Path, sandbox=None):
        self.root = Path(root).resolve()
        self.sandbox = sandbox or pick_sandbox()

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
        argv = self.sandbox.wrap(["bash", "-c", command], self.root)
        r = subprocess.run(argv, cwd=self.root, env=clean_env(), capture_output=True,
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
    def __init__(self, client, workspace: Workspace, permissions: Permissions | None = None,
                 model: str = MODEL, max_turns: int = MAX_TURNS, trace: bool = False, log=print):
        self.client, self.workspace, self.model = client, workspace, model
        # No human attached (tests, scripts): anything that would ask is denied.
        self.permissions = permissions or Permissions.load(workspace.root)
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
        # THE checkpoint. The model asked; the harness decides.
        decision = self.permissions.authorize(call.name, call.input)
        if self.trace:
            self._show(f"PERMISSION {call.name}: {decision.action}", {"reason": decision.reason})
        if decision.action != "allow":
            # A denial is still a tool_result. The model reads it and adapts.
            return {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                    "content": f"Not run: {decision.reason}."}
        out, is_error = self.workspace.run(call.name, call.input)
        if self.trace:
            self._show(f"TOOL {call.name}{'  (error)' if is_error else ''}", {"input": call.input, "output": out})
        return {"type": "tool_result", "tool_use_id": call.id, "content": out, "is_error": is_error}

    def _show(self, title: str, payload) -> None:
        self.log(f"\n===== {title} =====\n{json.dumps(to_json(payload), indent=2)}")


def ask_in_terminal(workspace: Workspace):
    """The approval prompt. Shows what will happen (a diff for edits), then asks."""
    def ask(tool: str, args: dict) -> tuple[str, str]:
        print(f"\n  {tool} wants to run: {subject_of(tool, args)}")
        if tool == "write_file":
            try:
                old = workspace.read_file(args["path"]).splitlines()
            except Exception:
                old = []
            diff = list(difflib.unified_diff(old, args["content"].splitlines(), "before", "after", lineterm=""))
            print("\n".join("    " + line for line in diff[:40]))
            if len(diff) > 40:
                print(f"    ... {len(diff) - 40} more lines")
        answer = input("  allow? [y]es / [a]lways this session / [n]o: ").strip().lower()
        if answer.startswith("a"):
            return "always", ""
        if answer.startswith("y"):
            return "yes", ""
        return "no", input("  tell the agent why, or press enter: ").strip()
    return ask


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="A minimal coding agent with permissions and a sandbox.")
    ap.add_argument("prompt", nargs="*", help="run one task and exit; omit for a REPL")
    ap.add_argument("--cwd", default=".", help="working directory the agent may edit")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--trace", action="store_true", help="print every request, response and tool call")
    ap.add_argument("--mode", choices=list(MODES), help="permission mode (default: from settings, else 'default')")
    ap.add_argument("--allow", action="append", default=[], metavar="RULE", help='e.g. "run_bash(pytest*)"')
    ap.add_argument("--deny", action="append", default=[], metavar="RULE", help='e.g. "write_file(*.lock)"')
    ap.add_argument("--no-sandbox", action="store_true", help="run bash directly on your machine")
    args = ap.parse_args(argv)

    import anthropic  # imported here so the tests run without the SDK or an API key
    workspace = Workspace(args.cwd, pick_sandbox(not args.no_sandbox))
    permissions = Permissions.load(workspace.root, args.mode, args.allow, args.deny, ask_in_terminal(workspace))
    agent = Agent(anthropic.Anthropic(), workspace, permissions, args.model, args.max_turns, args.trace)
    print(f"agent in {workspace.root} ({args.model})")
    print(f"permission mode: {permissions.mode} ({MODES[permissions.mode]}); sandbox: {workspace.sandbox.name}")
    if workspace.sandbox.name == "none":
        print("WARNING: no sandbox. Bash commands you approve run with your full access.")
    if args.prompt:
        print(agent.send(" ".join(args.prompt)))
        return
    print("Type 'exit' to quit.")
    while (line := input("\nyou> ").strip()) not in {"exit", "quit"}:
        if line:
            print(f"\nagent> {agent.send(line)}")


if __name__ == "__main__":
    main()
