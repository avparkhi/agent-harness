"""Step 6: run the tools on a remote machine, a Daytona sandbox.

The split:

    your machine                          the Daytona sandbox
    ------------------------------        ---------------------------------
    the agent loop                        the working directory
    your ANTHROPIC_API_KEY                read_file / write_file
    permissions and approvals             run_bash / search_files
    the durable session log               HARNESS.md (project memory)

Only tool calls cross the line, and they go through the same Permissions check as before.
The model's commands run on a machine of their own, so a bad `rm -rf` costs a sandbox,
not your laptop, and the key that pays for the model never enters the sandbox.

DaytonaWorkspace has the same methods as the local Workspace, so the Agent doesn't know
or care where its tools run. That's the whole trick: one interface, two backends.

Durability across machines: the session stores the sandbox's id. `--resume` reconnects to
the same sandbox (starting it again if it auto-stopped), so the files the agent wrote are
still there, even if the process on your machine died in between.
"""
from __future__ import annotations

import json
import posixpath
import shlex
from pathlib import Path

from agent import BASH_TIMEOUT, SKIP_DIRS, Workspace
from context import REMEMBERED, MEMORY_FILE, load_memory, with_note

AUTO_STOP_MINUTES = 30   # an idle sandbox stops (and stops costing) after this; its disk is kept

# search_files runs where the files are. Arguments go in as environment variables, never
# pasted into the code, so a pattern can't break out of the script.
SEARCH_SCRIPT = r"""
import fnmatch, json, os, re, sys
pattern, base, glob, limit = re.compile(os.environ["P"]), os.environ["B"], os.environ["G"], int(os.environ["L"])
skip, root, hits = set(json.loads(os.environ["S"])), os.environ["R"], []
for d, dirs, files in os.walk(base):
    dirs[:] = sorted(x for x in dirs if x not in skip)
    for f in sorted(files):
        if not fnmatch.fnmatch(f, glob):
            continue
        path = os.path.join(d, f)
        try:
            lines = open(path, encoding="utf-8").read().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = os.path.relpath(path, root)
        hits += [f"{rel}:{n}: {line.strip()}" for n, line in enumerate(lines, 1) if pattern.search(line)]
        if len(hits) >= limit:
            print("\n".join(hits[:limit]) + f"\n[stopped at {limit} matches; narrow the search]")
            sys.exit()
print("\n".join(hits) or "no matches")
"""


class RemoteError(Exception):
    pass


class DaytonaWorkspace(Workspace):
    """Workspace whose files and shell are inside a Daytona sandbox."""

    def __init__(self, sandbox, root: str | None = None):
        self.sandbox = sandbox
        self.root = posixpath.normpath(root or sandbox.get_work_dir())
        self.label = f"Daytona sandbox {sandbox.id}"
        self.shell_note = (f"Shell commands run on a remote Linux machine (a Daytona sandbox) that is yours "
                           f"alone. Work inside {self.root}.")

    def _resolve(self, path: str) -> str:
        # Paths are strings here: pathlib's resolve() would look at *this* machine's disk.
        # The check stops `../` escapes; a symlink inside the sandbox can still point out of
        # the working directory, which is acceptable because the sandbox is the boundary.
        p = posixpath.normpath(posixpath.join(self.root, path))
        if p != self.root and not p.startswith(self.root + "/"):
            raise PermissionError(f"{path} is outside the working directory")
        return p

    def _exec(self, command: str, cwd: str | None = None, env: dict | None = None):
        r = self.sandbox.process.exec(command, cwd=cwd or self.root, env=env, timeout=BASH_TIMEOUT)
        return r.exit_code, r.result or ""

    def read_file(self, path: str) -> str:
        data = self.sandbox.fs.download_file(self._resolve(path))
        if data is None:
            raise FileNotFoundError(path)
        return data.decode()

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        self._exec(f"mkdir -p {shlex.quote(posixpath.dirname(p))}")
        self.sandbox.fs.upload_file(content.encode(), p)
        return f"wrote {len(content)} characters to {path}"

    def run_bash(self, command: str) -> str:
        # 2>&1 so the model sees errors too; Daytona returns one output stream.
        code, out = self._exec(f"bash -c {shlex.quote(command)} 2>&1")
        return f"exit code: {code}\noutput:\n{out}"

    def search_files(self, pattern: str, path: str = ".", glob: str = "*", limit: int = 200) -> str:
        env = {"P": pattern, "B": self._resolve(path), "G": glob, "L": str(limit),
               "S": json.dumps(sorted(SKIP_DIRS)), "R": self.root}
        code, out = self._exec(f"python3 -c {shlex.quote(SEARCH_SCRIPT)}", env=env)
        if code != 0:
            raise RemoteError(out.strip().splitlines()[-1] if out.strip() else f"search failed ({code})")
        return out.rstrip("\n")

    def _read_or_none(self, path: str) -> str | None:
        try:
            return self.read_file(path)
        except Exception:
            return None

    def memory(self) -> str:
        return load_memory(self.root, read_project=self._read_or_none)

    def remember(self, note: str) -> str:
        self.write_file(MEMORY_FILE, with_note(self._read_or_none(MEMORY_FILE), note))
        return REMEMBERED


# --- creating and reconnecting -----------------------------------------------------------

def open_sandbox(daytona, sandbox_id: str | None = None, *, session_id: str | None = None,
                 offline: bool = False, log=print):
    """Reconnect to sandbox_id, or create a new sandbox. `daytona` is a daytona.Daytona client."""
    if sandbox_id:
        sandbox = daytona.get(sandbox_id)          # raises if the sandbox was deleted
        state = str(getattr(sandbox.state, "value", sandbox.state))
        if state != "started":
            log(f"[sandbox {sandbox_id} is {state}; starting it]")
            sandbox.start()
        return sandbox
    try:
        from daytona import CreateSandboxFromSnapshotParams as Params
    except ImportError:          # the tests use a fake client and may not have the SDK
        def Params(**kw):
            return kw
    params = Params(language="python", auto_stop_interval=AUTO_STOP_MINUTES,
                    labels={"harness-session": session_id or "none"},
                    network_block_all=offline or None)
    sandbox = daytona.create(params)
    log(f"[created Daytona sandbox {sandbox.id}]")
    return sandbox


def upload(workspace: DaytonaWorkspace, local_dir: str | Path) -> int:
    """Copy a local folder into the sandbox's working directory. Returns the file count."""
    local = Path(local_dir).resolve()
    files = [f for f in sorted(local.rglob("*"))
             if f.is_file() and not SKIP_DIRS & set(f.relative_to(local).parts)]
    folders = sorted({posixpath.dirname(workspace._resolve(f.relative_to(local).as_posix())) for f in files})
    if folders:
        workspace._exec("mkdir -p " + " ".join(shlex.quote(d) for d in folders))
    for f in files:
        workspace.sandbox.fs.upload_file(f.read_bytes(), workspace._resolve(f.relative_to(local).as_posix()))
    return len(files)


def download(workspace: DaytonaWorkspace, local_dir: str | Path) -> int:
    """Copy the sandbox's working directory to a local folder. Returns the file count."""
    prune = " -o ".join(f"-name {shlex.quote(d)}" for d in sorted(SKIP_DIRS))
    code, out = workspace._exec(f"find . \\( {prune} \\) -prune -o -type f -print")
    if code != 0:
        raise RemoteError(out)
    local = Path(local_dir).resolve()
    paths = [line[2:] if line.startswith("./") else line for line in out.splitlines() if line.strip()]
    for rel in paths:
        target = (local / rel).resolve()
        if not target.is_relative_to(local):     # never write outside the folder you named
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(workspace.sandbox.fs.download_file(workspace._resolve(rel)))
    return len(paths)
