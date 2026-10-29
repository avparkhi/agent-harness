# harness

A coding agent harness built from scratch, one layer at a time, to learn how tools like
Claude Code work inside. Each step is its own folder and git tag, so you can diff them.

| Step | Folder | What it adds |
|---|---|---|
| 1 | `step1_loop/` | The agent loop: model call, tool use, tool results, repeat. Three tools and a `--trace` flag. |
| 2 | | Permissions and sandboxing |
| 3 | | Context management: compaction, prompt caching, a memory file |
| 4 | | Subagents and planning |
| 5 | | Evals: a benchmark of real bugs |
| 6 | | A Duet-style coordinator, threads, tasks and routines |

## Setup

```
uv sync                       # or: pip install anthropic pytest
export ANTHROPIC_API_KEY=...  # only needed for live runs
cd step1_loop && python -m pytest -q
```

Start with `step1_loop/LESSON.md`.
