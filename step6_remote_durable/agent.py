"""mini-harness, step 6: step 4's agent, made durable and able to work on a remote machine.

New in this step:
    - durable.py: every step is committed to a SQLite log before it's shown or acted on, so a
      session survives a crash, a closed laptop or a kill -9, and `--resume` carries on
    - remote.py: the tools run in a Daytona sandbox instead of on your machine. The loop and
      your API key stay local; only file reads, writes and shell commands go to the sandbox.
Compare:  diff ../step4_subagents/agent.py agent.py

Run:  python agent.py --cwd ./scratch --durable "make a hello world script"
      python agent.py --cwd ./scratch --remote --durable --upload "run the tests and fix what fails"
      python agent.py --cwd ./scratch --list
      python agent.py --cwd ./scratch --resume SESSION_ID
"""
from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import re
import subprocess
import threading
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import context
from context import CONTINUATION, Meter, extract_summary, load_memory, with_summary_request
from durable import REPLAY_SAFE, LeaseHeld, Parked, Session, Store
from permissions import MODES, Permissions, subject_of
from planning import TODO_TOOL, check, render
from sandbox import clean_env, pick_sandbox
from subagents import TASK_TOOL, run_subagent

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
EFFORT = "high"          # Opus 5.5 defaults to "medium"; effort is how hard it thinks
MAX_TURNS = 25           # one turn = one model call; a cap stops runaway loops
BASH_TIMEOUT = 30        # seconds
MAX_RESULT_CHARS = 20_000  # long tool output is cut so it can't flood the context
COMPACT_AT = 200_000     # tokens; Opus 5.5 holds 1M, but long contexts cost more and focus less
MAX_PARALLEL = 4         # subagents running at once
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".harness"}

SYSTEM_PROMPT = """You are a coding agent working in the directory {cwd}.
Use the tools to read files, write files and run shell commands there.
Read a file before you change it. After changing code, run it or its tests to check your work.
Some tool calls need the user's approval. If one is denied, don't retry the same call:
read the reason, then change your approach or ask the user.
{shell}
When you learn something about this project that a future session should know (how to run
the tests, a convention, a pitfall), save it with the remember tool. Keep notes short.
For a task with three or more steps, write a todo list with todo_write before you start, and
keep it current as you go.
Use the task tool to hand work to subagents when it would flood your context (searching a
large codebase, reading many files) or when pieces are independent and can run in parallel.
A subagent can't see this conversation: give it a complete brief. Check its report before
you rely on it. Do small, quick things yourself.
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
    {
        "name": "remember",
        "description": "Save a short note to the project's memory file (HARNESS.md). Future sessions "
                       "see it in their system prompt. Use for durable facts, not task progress.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string", "description": "One line, written as a fact or rule."}},
            "required": ["note"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "search_files",
        "description": "Search file contents with a regular expression. Returns matching lines as "
                       "path:line: text (at most 200). Skips .git, virtualenvs and caches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": {"type": "string", "description": "Folder to search, relative to the working directory. Use . for all."},
                "glob": {"type": "string", "description": "Only files whose name matches, e.g. *.py. Use * for all."},
            },
            "required": ["pattern", "path", "glob"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    TODO_TOOL,
    TASK_TOOL,
]


class Workspace:
    """Runs tool calls. It no longer decides anything: by the time a call gets here,
    Permissions has already allowed it. Bash runs inside the sandbox."""

    def __init__(self, root: str | Path, sandbox=None):
        self.root = Path(root).resolve()
        self.sandbox = sandbox or pick_sandbox()
        self.label = f"this machine, sandbox: {self.sandbox.name}"
        self.shell_note = f"Shell commands run in a sandbox: no network, and writes only inside {self.root} and /tmp."

    def memory(self) -> str:
        return load_memory(self.root)

    def remember(self, note: str) -> str:
        return context.remember(self.root, note)

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

    def search_files(self, pattern: str, path: str = ".", glob: str = "*", limit: int = 200) -> str:
        regex = re.compile(pattern)
        base = self._resolve(path)
        hits = []
        for file in sorted(base.rglob("*")):
            rel = file.relative_to(self.root)
            if not file.is_file() or SKIP_DIRS & set(rel.parts) or not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                lines = file.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            hits += [f"{rel}:{n}: {line.strip()}" for n, line in enumerate(lines, 1) if regex.search(line)]
            if len(hits) >= limit:
                return "\n".join(hits[:limit]) + f"\n[stopped at {limit} matches; narrow the search]"
        return "\n".join(hits) or "no matches"

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (output, is_error). Errors go back to the model as text, not exceptions:
        a model that sees 'file not found' can recover; a crashed harness can't."""
        handler = {"read_file": self.read_file, "write_file": self.write_file, "run_bash": self.run_bash,
                   "search_files": self.search_files,
                   "remember": self.remember}.get(name)
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


METER_FIELDS = ("input", "cache_write", "cache_read", "output", "requests", "last_context", "subagent_cost")


class Agent:
    def __init__(self, client, workspace: Workspace, permissions: Permissions | None = None,
                 model: str = MODEL, max_turns: int = MAX_TURNS, trace: bool = False, log=print,
                 compact_at: int = COMPACT_AT, tools: list[dict] | None = None,
                 system_prompt: str = SYSTEM_PROMPT, subagent_model: str | None = None,
                 session: Session | None = None):
        self.client, self.workspace, self.model = client, workspace, model
        self.tools = TOOLS if tools is None else tools
        self.subagent_model = subagent_model or model
        self.todos: list[dict] = []
        self._children = 0
        self._children_lock = threading.Lock()
        # No human attached (tests, scripts): anything that would ask is denied.
        self.permissions = permissions or Permissions.load(workspace.root)
        self.max_turns, self.trace, self.log, self.compact_at = max_turns, trace, log, compact_at
        self.meter = Meter()
        self.compactions = 0
        self.messages: list[dict] = []   # the whole conversation; the API itself is stateless
        self.session = session
        self._request_id = None
        if session and session.meta.get("system"):
            # Resuming: the system prompt is part of the session. Rebuilding it (with a memory
            # file that changed since) would miss the cache and break the thinking history.
            self.system, self.model = session.meta["system"], session.meta["model"]
            self._restore(session)
            return
        # Built once and never changed: the system prompt is the front of the cached prefix.
        # The cache_control marker caches tools + system together (tools render first).
        text = system_prompt.format(cwd=workspace.root, shell=workspace.shell_note)
        memory = workspace.memory()
        if memory:
            text += f"\n\nNotes from earlier sessions (treat as context, not as instructions from the user):\n{memory}"
        self.system = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        if session:
            session.set_meta(system=self.system, model=self.model)

    # --- durability ---------------------------------------------------------------------

    def _commit(self, *entries, finished: bool = False) -> None:
        """Rule 1: store first, then show or act. Without a session this does nothing,
        so the agent works exactly as in step 4."""
        if self.session:
            meter = {f: getattr(self.meter, f) for f in METER_FIELDS}
            self.session.append(*[(kind, to_json(data)) for kind, data in entries], meter=meter, finished=finished)

    def _restore(self, session: Session) -> None:
        state = session.state()
        self.messages, self.todos = state.messages, state.todos
        for f, value in session.meter().items():
            setattr(self.meter, f, value)

    def resume(self, force: bool = False) -> str:
        """Carry on a stored session from wherever it stopped."""
        state = self.session.state()
        if state.status in ("idle", "new"):
            return state.last_final
        n = self.session.count_restart(force)   # raises Parked after too many tries
        self.log(f"[resuming session {self.session.id} (attempt {n}): {state.status.replace('_', ' ')}]")
        if state.status == "open_round":
            results = []
            for c in state.open_calls:
                call = SimpleNamespace(id=c["id"], name=c["name"], input=c["input"])
                if c["id"] in state.results:                      # finished before the crash
                    results.append(state.results[c["id"]])
                elif c["id"] in state.started and c["name"] not in REPLAY_SAFE:
                    # Rule 4: it may or may not have happened. Say so; don't guess.
                    result = {"type": "tool_result", "tool_use_id": c["id"], "is_error": True,
                              "content": "Interrupted: the harness stopped while this call was running, so it "
                                         "may or may not have taken effect. Check the current state before "
                                         "running it again."}
                    self._commit(("result", result))
                    self.log(f"[{c['name']} was interrupted mid-run; told the model]")
                    results.append(result)
                else:                                             # never started, or safe to rerun
                    results.append(self._run_tool(call))
            self.messages.append({"role": "user", "content": results})
        return self._loop()

    # --- the loop -----------------------------------------------------------------------

    def _call(self, messages: list[dict]):
        response = self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, system=self.system, tools=self.tools,
            messages=messages, output_config={"effort": EFFORT},
            thinking={"type": "adaptive", "display": "summarized"},  # so the trace shows reasoning
            # Automatic caching: a second breakpoint that follows the end of the conversation,
            # so each request reads everything the previous one wrote.
            cache_control={"type": "ephemeral"},
        )
        self.meter.add(response.usage)
        return response

    def send(self, user_text: str, request_id: str | None = None) -> str:
        """request_id makes a submit exactly-once: send the same id twice (a retry after a
        network error, say) and the second call returns the first answer instead of redoing it."""
        if self.session and request_id:
            state = self.session.state()
            if request_id in state.answers:
                return state.answers[request_id]
            if request_id in state.requests:
                return self.resume()
        self._request_id = request_id
        self._commit(("user", {"content": user_text, "request_id": request_id}))
        self.messages.append({"role": "user", "content": user_text})
        return self._loop()

    def _finish(self, assistant: dict | None, text: str) -> str:
        entries = ([("assistant", assistant)] if assistant else []) + [("final", {"text": text})]
        self._commit(*entries, finished=True)   # the answer and "done" land together, or neither does
        if assistant:
            self.messages.append(assistant)
        return text

    def _loop(self) -> str:
        for turn in range(1, self.max_turns + 1):
            # Safe point: the last message is a user turn, so no tool call is waiting on a result.
            if self.meter.last_context > self.compact_at:
                self.compact()
            if self.trace:
                self._show(f"REQUEST {turn}", {"system": self.system if self.meter.requests == 0 else "(unchanged)",
                                               "tools": [t["name"] for t in self.tools], "messages": self.messages})
            # A crash during this call loses nothing: the request is rebuilt from the log and resent.
            response = self._call(self.messages)
            # The assistant turn exactly as it came back, thinking blocks included.
            assistant = {"role": "assistant", "content": response.content}
            if self.trace:
                self._show(f"RESPONSE {turn}  stop_reason={response.stop_reason}",
                           {"content": response.content, "usage": response.usage})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if not tool_calls:
                # end_turn (done), or max_tokens/refusal with no pending tool call: hand back to the user.
                text = "".join(b.text for b in response.content if b.type == "text")
                return self._finish(assistant, text if response.stop_reason == "end_turn"
                                    else f"{text}\n[stopped: {response.stop_reason}]")
            self._commit(("assistant", {"content": response.content, "stop_reason": response.stop_reason}))
            self.messages.append(assistant)
            if response.stop_reason == "tool_use":
                results = self._run_tools(tool_calls)
            else:
                # e.g. max_tokens cut the turn off mid-call. Every tool_use still needs a
                # tool_result, or the next request is rejected, so answer them with errors.
                results = [{"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                            "content": f"not run: your turn was cut off (stop_reason={response.stop_reason})"}
                           for c in tool_calls]
                self._commit(*[("result", r) for r in results])
            # Tool results go back as a *user* turn: only the user side can carry tool_result blocks.
            self.messages.append({"role": "user", "content": results})
        return self._finish(None, f"[stopped: hit the {self.max_turns}-turn cap]")

    def compact(self) -> str:
        """Simple compaction: summarize the whole conversation, then replace it with one user
        turn holding the summary (plus whatever was waiting to be answered). Nothing else is
        replayed: no old turns and no old thinking blocks, because thinking blocks are only
        valid next to the exact history that produced them."""
        before = self.meter.last_context
        response = self._call(with_summary_request(self.messages))
        summary = extract_summary(response.content)
        if not summary:
            self.log("[compaction failed: no summary came back; keeping the full history]")
            return ""
        pending = self.messages[-1]
        intro_text = CONTINUATION.format(summary=summary)
        if self.todos:   # the plan is harness state, so it survives compaction exactly
            intro_text += f"Your todo list:\n{render(self.todos)}\n\n"
        intro = {"type": "text", "text": intro_text}
        if pending["role"] == "user" and isinstance(pending["content"], str):
            content = [intro, {"type": "text", "text": pending["content"]}]        # a new user message
        elif pending["role"] == "user":
            # Mid-task: the tool_use these results answered is gone, so pass them on as text.
            done = "\n\n".join(f"Result of your last tool call:\n{r['content']}" for r in pending["content"])
            content = [intro, {"type": "text", "text": f"{done}\n\nContinue the task from where it stopped."}]
        else:
            content = [intro, {"type": "text", "text": "(Waiting for the user's next message.)"}]
        compacted = [{"role": "user", "content": content}]
        self.meter.last_context = 0
        # The log keeps every old entry; this one says "from here on, the conversation is this".
        self._commit(("compaction", {"messages": compacted}))
        self.messages = compacted
        self.compactions += 1
        self.log(f"[compacted: about {before:,} tokens of conversation replaced by a "
                 f"{len(summary):,}-character summary]")
        if self.trace:
            self._show("COMPACTION SUMMARY", {"summary": summary})
        return summary

    def _run_tools(self, calls) -> list[dict]:
        """Subagent calls in the same turn run in parallel; every other call runs in order.
        Results go back in the order the model asked, all in one user turn."""
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            running = {c.id: pool.submit(self._run_tool, c) for c in calls if c.name == "task"}
            return [running[c.id].result() if c.id in running else self._run_tool(c) for c in calls]

    def next_child_number(self) -> int:
        with self._children_lock:
            self._children += 1
            return self._children

    def _todo_write(self, todos: list[dict]) -> tuple[str, bool]:
        problem = check(todos)
        if problem:
            return f"Todo list not saved: {problem}.", True
        self.todos = todos
        return (f"Todo list saved:\n{render(todos)}\n"
                "Work through it in order and mark items completed as you finish them."), False

    def _run_tool(self, call) -> dict:
        # THE checkpoint. The model asked; the harness decides.
        decision = self.permissions.authorize(call.name, call.input)
        if self.trace:
            self._show(f"PERMISSION {call.name}: {decision.action}", {"reason": decision.reason})
        if decision.action != "allow":
            # A denial is still a tool_result. The model reads it and adapts.
            result = {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                      "content": f"Not run: {decision.reason}."}
            self._commit(("result", result))
            return result
        # Rule 3: intent before effect. From here on, a crash means "this may have happened".
        self._commit(("intent", {"id": call.id, "name": call.name, "input": call.input}))
        extra = []
        if call.name == "todo_write":
            out, is_error = self._todo_write(call.input["todos"])
            extra = [("todos", {"todos": self.todos})]
        elif call.name == "task":
            out, is_error = run_subagent(self, TOOLS, **call.input)
        else:
            out, is_error = self.workspace.run(call.name, call.input)
        result = {"type": "tool_result", "tool_use_id": call.id, "content": out, "is_error": is_error}
        self._commit(("result", result), *extra)   # the result and the plan it set land together
        if extra and not is_error:
            self.log(f"\n[plan]\n{render(self.todos)}")
        if self.trace:
            self._show(f"TOOL {call.name}{'  (error)' if is_error else ''}", {"input": call.input, "output": out})
        return result

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


def show_session(store: Store, sid: str) -> None:
    """Print a session's log, one entry per line: what it did, in order."""
    for kind, data in store.entries(sid):
        if kind in ("user", "final"):
            body = data.get("content", data.get("text"))
            line = body if isinstance(body, str) else json.dumps(body)
        elif kind == "assistant":
            line = " + ".join(b.get("text", "")[:60] or f"{b['type']} {b.get('name', '')}".strip()
                              for b in data["content"] if b["type"] in ("text", "tool_use"))
        elif kind == "intent":
            line = f"{data['name']} {json.dumps(data['input'])}"
        elif kind == "result":
            line = ("ERROR " if data["is_error"] else "") + str(data["content"]).replace("\n", " / ")
        else:
            line = json.dumps(data)
        print(f"  {kind:10} {line[:110]}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="A coding agent with permissions, memory, planning, subagents, "
                                             "durable sessions and remote sandboxes.")
    ap.add_argument("prompt", nargs="*", help="run one task and exit; omit for a REPL")
    ap.add_argument("--cwd", default=".", help="your project folder (settings, sessions; the workspace unless --remote)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--trace", action="store_true", help="print every request, response and tool call")
    ap.add_argument("--mode", choices=list(MODES), help="permission mode (default: from settings, else 'default')")
    ap.add_argument("--allow", action="append", default=[], metavar="RULE", help='e.g. "run_bash(pytest*)"')
    ap.add_argument("--deny", action="append", default=[], metavar="RULE", help='e.g. "write_file(*.lock)"')
    ap.add_argument("--no-sandbox", action="store_true", help="run bash directly on your machine")
    ap.add_argument("--subagent-model", help="model for subagents (default: same as --model)")
    ap.add_argument("--compact-at", type=int, default=COMPACT_AT, metavar="TOKENS",
                    help=f"summarize the conversation once it passes this size (default {COMPACT_AT:,})")
    durable = ap.add_argument_group("durable sessions")
    durable.add_argument("--durable", action="store_true", help="store the session so it survives a crash")
    durable.add_argument("--db", help="session file (default: CWD/.harness/sessions.db)")
    durable.add_argument("--resume", metavar="ID", help="carry on a stored session")
    durable.add_argument("--force", action="store_true", help="resume a session parked after repeated crashes")
    durable.add_argument("--list", action="store_true", help="list stored sessions")
    durable.add_argument("--show", metavar="ID", help="print a stored session's log")
    durable.add_argument("--request-id", help="send the prompt at most once, even if you run this command twice")
    remote = ap.add_argument_group("remote sandbox (Daytona; needs DAYTONA_API_KEY)")
    remote.add_argument("--remote", action="store_true", help="run the tools in a Daytona sandbox")
    remote.add_argument("--upload", action="store_true", help="copy CWD into the sandbox first")
    remote.add_argument("--download", metavar="DIR", help="copy the sandbox's files to DIR at the end")
    remote.add_argument("--offline-sandbox", action="store_true", help="block the sandbox's network")
    remote.add_argument("--keep-sandbox", action="store_true",
                        help="keep the sandbox of a non-durable run (durable runs always keep it)")
    args = ap.parse_args(argv)

    local_root = Path(args.cwd).resolve()
    store = None
    if args.durable or args.resume or args.list or args.show:
        db = Path(args.db) if args.db else local_root / ".harness" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        store = Store(db)
    if args.list:
        for s in store.list():
            where = f"remote {s['meta']['sandbox_id']}" if s["meta"].get("sandbox_id") else "local"
            print(f"{s['id']}  {s['status']:11} {'(running) ' if s['live'] else ''}{where:22} {s['first_prompt'][:60]}")
        return
    if args.show:
        show_session(store, args.show)
        return

    session = None
    try:
        if args.resume:
            session = Session.open(store, args.resume)
        elif args.durable:
            session = Session.create(store, {"cwd": str(local_root)})
    except LeaseHeld as e:
        raise SystemExit(f"{e}. Wait for it, or stop that process first.")
    meta = session.meta if session else {}

    daytona = sandbox = None
    if args.remote or meta.get("sandbox_id"):
        try:
            from daytona import Daytona
        except ImportError:
            raise SystemExit("--remote needs the Daytona SDK: python -m pip install daytona")
        from remote import DaytonaWorkspace, download, open_sandbox, upload
        try:
            daytona = Daytona()   # reads DAYTONA_API_KEY from the environment; never put the key in code
            sandbox = open_sandbox(daytona, meta.get("sandbox_id"), session_id=session and session.id,
                                   offline=args.offline_sandbox)
        except Exception as e:
            if session and not args.resume:     # nothing happened yet: don't leave an empty session
                store.tx(lambda db: db.execute("delete from sessions where id = ?", (session.id,)))
            elif session:
                session.close()
            raise SystemExit(f"couldn't reach the Daytona sandbox: {e}")
        if session and not meta.get("sandbox_id"):
            session.set_meta(sandbox_id=sandbox.id)
        workspace = DaytonaWorkspace(sandbox)
        if args.upload:
            print(f"[uploaded {upload(workspace, local_root)} files to {workspace.root}]")
    else:
        workspace = Workspace(local_root, pick_sandbox(not args.no_sandbox))

    import anthropic  # imported here so the tests run without the SDK or an API key
    # Settings and approvals come from your local project, wherever the tools run.
    permissions = Permissions.load(local_root, args.mode, args.allow, args.deny, ask_in_terminal(workspace))
    agent = Agent(anthropic.Anthropic(), workspace, permissions, args.model, args.max_turns, args.trace,
                  compact_at=args.compact_at, subagent_model=args.subagent_model, session=session)
    print(f"agent in {workspace.root} ({agent.model}) on {workspace.label}")
    print(f"permission mode: {permissions.mode} ({MODES[permissions.mode]})")
    if getattr(workspace.sandbox, "name", None) == "none":
        print("WARNING: no sandbox. Bash commands you approve run with your full access.")
    if session:
        print(f"session: {session.id} (durable)")
        session.heartbeat()
    try:
        if args.resume:
            print(f"\nagent> {agent.resume(force=args.force)}")
        if args.prompt:
            print(agent.send(" ".join(args.prompt), request_id=args.request_id))
            print(f"\n[{agent.meter.summary()}]")
        elif not args.resume or agent.messages:
            print("Type 'exit' to quit, '/compact' to summarize now, '/cost' for usage.")
            while (line := input("\nyou> ").strip()) not in {"exit", "quit"}:
                if line == "/compact":
                    agent.compact() if agent.messages else print("nothing to compact yet")
                elif line == "/cost":
                    print(agent.meter.summary())
                elif line:
                    print(f"\nagent> {agent.send(line)}")
                    print(f"\n[{agent.meter.summary()}]")
    except Parked as e:
        print(f"\n{e}")
    except (KeyboardInterrupt, EOFError):
        print("\n[stopped]")
    finally:
        if args.download and sandbox is not None:
            print(f"[downloaded {download(workspace, args.download)} files to {args.download}]")
        if session:
            session.close()
            print(f"[session {session.id} saved. Carry on with: --resume {session.id}]")
        elif sandbox is not None and not args.keep_sandbox:
            daytona.delete(sandbox)
            print(f"[deleted sandbox {sandbox.id}]")


if __name__ == "__main__":
    main()
