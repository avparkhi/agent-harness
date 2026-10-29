"""Step 2: run bash inside an OS sandbox.

Permission rules are guesses about what a command string will do. A sandbox doesn't guess:
the operating system itself refuses. Rules decide *whether* to run; the sandbox limits
*what can happen* if a rule was wrong. You want both.

Inside the sandbox a command can:
    - read most of the filesystem, except well-known secret folders (they look empty)
    - write only inside the working directory and a private /tmp
    - not use the network
and it gets a cleaned environment, so API keys in your shell aren't visible to it.

Backends: bubblewrap on Linux (`apt install bubblewrap`), sandbox-exec (Seatbelt) on macOS.
"""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

SECRET_PATHS = ["~/.ssh", "~/.aws", "~/.config/gcloud", "~/.azure", "~/.config/anthropic",
                "~/.docker", "~/.kube", "~/.netrc", "~/.git-credentials", "~/.npmrc", "~/.pypirc"]

# Only these variables reach the command. Everything else (ANTHROPIC_API_KEY, AWS_*,
# GITHUB_TOKEN ...) is dropped, or the model could just run `env` and read them.
SAFE_ENV = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR", "VIRTUAL_ENV"}


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in SAFE_ENV}


def existing_secrets() -> list[Path]:
    paths = [Path(p).expanduser() for p in SECRET_PATHS]
    return [p.resolve() for p in paths if p.exists()]


class NoSandbox:
    name = "none"

    def wrap(self, argv: list[str], cwd: Path) -> list[str]:
        return argv


class Bubblewrap:
    """Linux. Builds a private view of the filesystem with mount namespaces."""
    name = "bubblewrap"

    def wrap(self, argv: list[str], cwd: Path) -> list[str]:
        cmd = ["bwrap",
               "--ro-bind", "/", "/",          # the whole filesystem, read-only
               "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp"]              # a fresh, private /tmp
        for secret in existing_secrets():      # hide secrets behind empty stand-ins
            cmd += ["--tmpfs", str(secret)] if secret.is_dir() else ["--ro-bind", "/dev/null", str(secret)]
        cmd += ["--bind", str(cwd), str(cwd),  # the one place it may write
                "--unshare-net",               # no network
                "--die-with-parent", "--chdir", str(cwd), "--"]
        return cmd + argv


class Seatbelt:
    """macOS. Same idea through the built-in sandbox-exec and a policy profile.
    Not tested in the Linux container this was built in; try it on your Mac."""
    name = "seatbelt"

    def wrap(self, argv: list[str], cwd: Path) -> list[str]:
        secrets = " ".join(f'(subpath "{p}")' for p in existing_secrets())
        profile = "\n".join([
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny file-write*)",
            '(allow file-write* (subpath (param "CWD")) (subpath "/private/tmp")'
            ' (subpath "/private/var/folders") (literal "/dev/null") (literal "/dev/tty"))',
            f"(deny file-read* {secrets})" if secrets else "",
        ])
        return ["sandbox-exec", "-p", profile, "-D", f"CWD={cwd}", *argv]


def pick_sandbox(enabled: bool = True):
    if not enabled:
        return NoSandbox()
    if platform.system() == "Linux" and shutil.which("bwrap"):
        return Bubblewrap()
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return Seatbelt()
    return NoSandbox()
