"""Sandbox tests. They run real commands, so the bubblewrap ones only run where bwrap exists."""
import shutil
from pathlib import Path

import pytest

import sandbox
from agent import Workspace
from sandbox import Bubblewrap, NoSandbox, Seatbelt, clean_env

needs_bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not installed")


def test_clean_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    env = clean_env()
    assert "ANTHROPIC_API_KEY" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    assert "PATH" in env


def test_bash_does_not_see_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    out, _ = Workspace(tmp_path, NoSandbox()).run("run_bash", {"command": "env"})
    assert "sk-ant-secret" not in out


def test_seatbelt_profile_shape(tmp_path):
    argv = Seatbelt().wrap(["bash", "-c", "true"], tmp_path)
    assert argv[0] == "sandbox-exec" and f"CWD={tmp_path}" in argv
    assert "(deny network*)" in argv[2] and "(deny file-write*)" in argv[2]


@needs_bwrap
def test_writes_inside_cwd_work(tmp_path):
    out, is_error = Workspace(tmp_path, Bubblewrap()).run("run_bash", {"command": "echo hi > a.txt && cat a.txt"})
    assert not is_error and "exit code: 0" in out
    assert (tmp_path / "a.txt").read_text() == "hi\n"


@needs_bwrap
def test_writes_outside_cwd_are_blocked(tmp_path):
    outside = Path("/var/tmp") / f"{tmp_path.name}-outside.txt"
    out, _ = Workspace(tmp_path, Bubblewrap()).run("run_bash", {"command": f"echo x > {outside}"})
    assert "Read-only file system" in out
    assert not outside.exists()


@needs_bwrap
def test_tmp_is_private(tmp_path):
    # /tmp inside the sandbox is a fresh in-memory folder: writes there never reach your disk.
    leak = tmp_path.parent / f"{tmp_path.name}-leak.txt"
    out, _ = Workspace(tmp_path, Bubblewrap()).run("run_bash", {"command": f"echo x > {leak}"})
    assert "exit code: 0" in out
    assert not leak.exists()


@needs_bwrap
def test_network_is_blocked(tmp_path):
    out, _ = Workspace(tmp_path, Bubblewrap()).run(
        "run_bash", {"command": "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), 3)\""})
    assert "exit code: 0" not in out


@needs_bwrap
def test_secret_folders_look_empty(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh/id_ed25519").write_text("PRIVATE KEY")
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "work"
    work.mkdir()
    out, _ = Workspace(work, Bubblewrap()).run("run_bash", {"command": f"cat {home}/.ssh/id_ed25519; ls -A {home}/.ssh"})
    assert "PRIVATE KEY" not in out


def test_pick_sandbox_off():
    assert sandbox.pick_sandbox(enabled=False).name == "none"
