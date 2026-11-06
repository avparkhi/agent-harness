"""Step 4a: a todo list the model plans with.

The harness doesn't plan. The model does, and the todo_write tool gives the plan a place
to live: every call sends the *whole* list, the harness checks and shows it, and the list
comes back in the tool result. Because it's in the conversation, the model re-reads its
own plan on every turn, which keeps long tasks on track. This is how Claude Code's
TodoWrite works.
"""
from __future__ import annotations

STATUSES = ("pending", "in_progress", "completed")

TODO_TOOL = {
    "name": "todo_write",
    "description": "Create or update your todo list for the current task. Send the complete list "
                   "every time (it replaces the old one). Use it for any task with 3 or more steps. "
                   "Keep exactly one item in_progress while you work, and mark each item completed "
                   "as soon as it's done, not in a batch at the end.",
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to do, as a short imperative."},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    "required": ["content", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["todos"],
        "additionalProperties": False,
    },
    "strict": True,
}

MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


def check(todos: list[dict]) -> str | None:
    """Returns a problem to send back to the model, or None if the list is fine."""
    if any(not t.get("content", "").strip() for t in todos):
        return "every todo needs content"
    if any(t.get("status") not in STATUSES for t in todos):
        return f"status must be one of {', '.join(STATUSES)}"
    if sum(t["status"] == "in_progress" for t in todos) > 1:
        return "only one todo can be in_progress at a time; finish or pause the others"
    return None


def render(todos: list[dict]) -> str:
    return "\n".join(f"{MARKS[t['status']]} {t['content']}" for t in todos) or "(empty)"
