"""Step 4b: subagents.

A subagent is the same loop as the main agent, started fresh: its own empty conversation,
its own system prompt, a smaller set of tools. The main agent hands it a brief with the
`task` tool; the subagent works (maybe 20 tool calls) and returns one final report, which
becomes the tool_result. The parent never sees the subagent's file reads and dead ends.

That is the whole point: context isolation. The parent's context stays small and on topic,
and independent subagents can run in parallel. The price: the subagent knows only what's
in its brief, and the parent knows only what's in the report.
"""
from __future__ import annotations

import textwrap

SUBAGENT_MAX_TURNS = 15

TASK_TOOL = {
    "name": "task",
    "description": "Hand a self-contained job to a subagent. It starts with an empty context: it cannot "
                   "see this conversation, so the prompt must include everything it needs (goal, files, "
                   "constraints, what to report back). Its final reply comes back as this tool's result. "
                   "Use 'explore' (read-only, fast) for searching and reading code; use 'general' for "
                   "independent pieces of work that change files. Call task several times in one turn "
                   "to run independent subagents in parallel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "3 to 5 words, shown to the user."},
            "prompt": {"type": "string", "description": "The complete brief for the subagent."},
            "agent_type": {"type": "string", "enum": ["explore", "general"]},
        },
        "required": ["description", "prompt", "agent_type"],
        "additionalProperties": False,
    },
    "strict": True,
}

_REPORT = """Your final reply is the only thing the agent that started you will see, so make
it a complete report: what you found or did, with file paths and line numbers, and anything
you were unsure about. Don't include your process unless it matters."""

SUBAGENT_TYPES = {
    "explore": {
        "tools": ["read_file", "search_files"],
        "mode": "read-only",
        "prompt": "You are a read-only research subagent working in {cwd}. Another agent gave you a "
                  "question about this codebase. Search and read files to answer it; you cannot change "
                  "anything. Be quick: search first, then read only what you need.\n" + _REPORT,
    },
    "general": {
        "tools": ["read_file", "write_file", "run_bash", "search_files"],
        "mode": None,   # same permissions as the parent
        "prompt": "You are a subagent working in {cwd}. Another agent gave you one self-contained job. "
                  "Do it, check your work (run it or its tests), and stop. Stay inside the job you were "
                  "given; if it can't be done as described, say why instead of improvising.\n"
                  "{shell}\n"
                  + _REPORT,
    },
}


def run_subagent(parent, catalog: list[dict], description: str, prompt: str, agent_type: str) -> tuple[str, bool]:
    """Run one subagent to completion and return (report, is_error)."""
    spec = SUBAGENT_TYPES.get(agent_type)
    if spec is None:
        return f"unknown agent_type {agent_type!r}; use explore or general", True
    name = f"{agent_type}-{parent.next_child_number()}"
    permissions = parent.permissions if spec["mode"] is None else parent.permissions.with_mode(spec["mode"])

    def log(message: str) -> None:
        parent.log(textwrap.indent(str(message), f"  [{name}] "))

    child = type(parent)(
        parent.client, parent.workspace, permissions,
        model=parent.subagent_model, max_turns=SUBAGENT_MAX_TURNS, trace=parent.trace, log=log,
        compact_at=parent.compact_at, system_prompt=spec["prompt"],
        tools=[t for t in catalog if t["name"] in spec["tools"]],   # no `task`: subagents can't spawn more
    )
    parent.log(f"[{name} started: {description}]")
    report = child.send(prompt)
    parent.meter.absorb(child.meter)
    parent.log(f"[{name} finished: {child.meter.requests} requests, ${child.meter.cost:.4f}]")
    stopped = not report or report.splitlines()[-1].startswith("[stopped")
    return report, stopped
