"""Step 2: decide whether a tool call is allowed to run.

The model only *requests* tool calls. This module sits between the request and the
handler and answers one of three things for every call:
    allow  run it now
    deny   don't run it; tell the model why
    ask    stop and ask the human

The order of checks is the whole design:
    1. deny rules      always win, in every mode
    2. read-only mode  anything that isn't a read is denied
    3. allow rules     run without asking (for bash, every part of the command must match)
    4. the mode        decides the rest
    5. otherwise       ask
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

READ_TOOLS = {"read_file"}
EDIT_TOOLS = {"write_file"}
MODES = {
    "default": "reads run; edits and bash ask first",
    "accept-edits": "reads and edits run; bash asks first",
    "read-only": "only reads; everything else is denied (like a planning mode)",
    "bypass": "everything runs without asking; deny rules and the sandbox still apply",
}
SETTINGS_FILE = ".harness/settings.json"

# Rules that ship with the harness. The last deny matters most: without it the agent could
# write its own settings file and grant itself permission to do anything next session.
DEFAULT_DENY = ["run_bash(sudo *)", "read_file(.env)", "read_file(*/.env)",
                "write_file(.git/*)", "write_file(.harness/*)"]
DEFAULT_ALLOW = ["run_bash(ls)", "run_bash(ls *)", "run_bash(pwd)", "run_bash(cat *)",
                 "run_bash(git status*)", "run_bash(git diff*)", "run_bash(git log*)"]


@dataclass(frozen=True)
class Rule:
    """`tool(pattern)`, e.g. `run_bash(git status*)` or `write_file(src/*)`. A bare tool
    name matches every call to that tool. Patterns are shell-style globs, and `*` also
    matches `/`."""
    tool: str
    pattern: str = "*"

    @classmethod
    def parse(cls, text: str) -> Rule:
        text = text.strip()
        if text.endswith(")") and "(" in text:
            tool, pattern = text[:-1].split("(", 1)
            return cls(tool.strip(), pattern)
        return cls(text)

    def matches(self, tool: str, subject: str) -> bool:
        return tool == self.tool and fnmatch(subject, self.pattern)

    def __str__(self) -> str:
        return f"{self.tool}({self.pattern})"


@dataclass
class Decision:
    action: str   # "allow" | "deny" | "ask"
    reason: str


def subject_of(tool: str, args: dict) -> str:
    """The part of a call that rules match against: the command for bash, the path otherwise."""
    return args.get("command") if tool == "run_bash" else str(args.get("path", ""))


SEPARATORS = {"&&", "||", ";", "|", "&"}


def split_command(command: str) -> list[str] | None:
    """Split `git add . && git commit -m 'x | y'` into ['git add .', "git commit -m 'x | y'"].

    Without this, an allow rule `git status*` would approve `git status && rm -rf ~`.
    Returns None for anything we can't reason about (substitution, redirection, subshells),
    so those always fall through to asking. Being conservative here is the point."""
    if "`" in command or "$(" in command or "\n" in command:
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:            # unbalanced quotes
        return None
    parts, current = [], []
    for tok in tokens:
        if tok in SEPARATORS:
            parts.append(current)
            current = []
        elif tok and set(tok) <= set("();<>|&"):
            return None           # >, <, >>, (, ) ... redirection or subshell
        else:
            current.append(tok)
    parts.append(current)
    if any(not p for p in parts):
        return None
    return [shlex.join(p) for p in parts]


class Permissions:
    def __init__(self, mode: str = "default", allow=(), deny=(), ask_user=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
        self.mode = mode
        self.allow = [r if isinstance(r, Rule) else Rule.parse(r) for r in allow]
        self.deny = [r if isinstance(r, Rule) else Rule.parse(r) for r in deny]
        self.ask_user = ask_user    # callable(tool, args) -> (answer, feedback); None = no human
        self.approved: set[tuple[str, str]] = set()   # "always" answers, kept for this session only

    @classmethod
    def load(cls, root: Path, mode: str | None = None, allow=(), deny=(), ask_user=None) -> Permissions:
        """Defaults, then the project's settings file, then command-line flags."""
        settings = {}
        path = Path(root) / SETTINGS_FILE
        if path.exists():
            settings = json.loads(path.read_text()).get("permissions", {})
        return cls(mode or settings.get("defaultMode", "default"),
                   [*DEFAULT_ALLOW, *settings.get("allow", []), *allow],
                   [*DEFAULT_DENY, *settings.get("deny", []), *deny], ask_user)

    def check(self, tool: str, args: dict) -> Decision:
        subject = subject_of(tool, args)
        parts = split_command(subject) if tool == "run_bash" else [subject]

        for part in [subject, *(parts or [])]:
            for rule in self.deny:
                if rule.matches(tool, part):
                    return Decision("deny", f"blocked by deny rule {rule}")
        if self.mode == "read-only" and tool not in READ_TOOLS:
            return Decision("deny", "the session is in read-only mode")
        if (tool, subject) in self.approved:
            return Decision("allow", "approved earlier this session")
        if parts and all(any(r.matches(tool, p) for r in self.allow) for p in parts):
            return Decision("allow", "matched allow rules")
        if tool in READ_TOOLS:
            return Decision("allow", "reads are allowed")
        if self.mode == "bypass":
            return Decision("allow", "bypass mode")
        if self.mode == "accept-edits" and tool in EDIT_TOOLS:
            return Decision("allow", "accept-edits mode")
        return Decision("ask", "needs approval")

    def authorize(self, tool: str, args: dict) -> Decision:
        """check(), then resolve an "ask" by asking the human."""
        decision = self.check(tool, args)
        if decision.action != "ask":
            return decision
        if self.ask_user is None:
            return Decision("deny", "needs approval and no one is available to approve it")
        answer, feedback = self.ask_user(tool, args)
        if answer == "always":
            # Remember this exact call for the rest of the session (not saved to disk).
            self.approved.add((tool, subject_of(tool, args)))
            return Decision("allow", "approved by the user for this session")
        if answer == "yes":
            return Decision("allow", "approved by the user")
        return Decision("deny", "the user declined" + (f": {feedback}" if feedback else ""))
