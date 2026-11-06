"""A stand-in for the Daytona SDK, so the remote code runs with no account or API key.

It has the few calls remote.py uses (create, get, delete, sandbox.fs, sandbox.process.exec,
sandbox.start) and keeps each "sandbox" in a local folder. Paths under /home/daytona are
mapped into that folder, and commands run with plain bash. It is not isolated: it's for
tests and the offline demo, not for running untrusted code.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

WORK_DIR = "/home/daytona"


class FakeSandbox:
    def __init__(self, disk: Path, params, sandbox_id: str):
        self.id = sandbox_id
        self.params = params
        self.state = "started"
        self.starts = 0
        self.disk = disk
        (disk / WORK_DIR.lstrip("/")).mkdir(parents=True, exist_ok=True)
        self.fs = SimpleNamespace(upload_file=self._upload, download_file=self._download)
        self.process = SimpleNamespace(exec=self._exec)

    def get_work_dir(self) -> str:
        return WORK_DIR

    def start(self) -> None:
        self.state, self.starts = "started", self.starts + 1

    def stop(self) -> None:
        self.state = "stopped"

    def _local(self, text: str) -> str:
        return text.replace(WORK_DIR, str(self.disk / WORK_DIR.lstrip("/")))

    def _upload(self, data: bytes, path: str) -> None:
        self._check()
        Path(self._local(path)).write_bytes(data)

    def _download(self, path: str) -> bytes:
        self._check()
        return Path(self._local(path)).read_bytes()

    def _exec(self, command: str, cwd: str | None = None, env: dict | None = None, timeout: int | None = None):
        self._check()
        env = {k: self._local(v) for k, v in (env or {}).items()}
        r = subprocess.run(["bash", "-c", self._local(command)], cwd=self._local(cwd or WORK_DIR),
                           env={**os.environ, **env}, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).replace(str(self.disk / WORK_DIR.lstrip("/")), WORK_DIR)
        return SimpleNamespace(exit_code=r.returncode, result=out)

    def _check(self) -> None:
        if self.state != "started":
            raise RuntimeError(f"sandbox {self.id} is {self.state}")


class FakeDaytona:
    """Sandboxes live in folders under root, named by id, so a second process can get() a
    sandbox the first one created, just as a real one outlives the process that made it."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.sandboxes: dict[str, FakeSandbox] = {}

    def create(self, params=None, **_):
        sandbox_id = f"sbx-{uuid.uuid4().hex[:8]}"
        sandbox = FakeSandbox(self.root / sandbox_id, params, sandbox_id)
        self.sandboxes[sandbox.id] = sandbox
        return sandbox

    def get(self, sandbox_id: str) -> FakeSandbox:
        if sandbox_id not in self.sandboxes:
            if not (self.root / sandbox_id).is_dir():
                raise LookupError(f"sandbox {sandbox_id} not found")
            self.sandboxes[sandbox_id] = FakeSandbox(self.root / sandbox_id, None, sandbox_id)
        return self.sandboxes[sandbox_id]

    def delete(self, sandbox: FakeSandbox, **_) -> None:
        self.sandboxes.pop(sandbox.id, None)
