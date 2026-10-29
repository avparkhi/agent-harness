# harness

A coding agent harness built from scratch, one layer at a time, to learn how tools like
Claude Code work inside. Each step is its own folder, so you can diff one against the next.

| Step | Folder | What it adds |
|---|---|---|
| 1 | `step1_loop/` | The agent loop: model call, tool use, tool results, repeat. Three tools and a `--trace` flag. |
| 2 | `step2_permissions/` | Permission rules, modes and an approval prompt; bash runs in an OS sandbox. |
| 3 | `step3_context/` | A HARNESS.md memory file, prompt caching with a cost meter, and compaction. |
| 4 | `step4_subagents/` | A todo list the model plans with, and subagents with their own context, run in parallel. |
| 5 | | Evals: a benchmark of real bugs |
| 6 | | A Duet-style coordinator, threads, tasks and routines |

## Setup

```
uv sync                       # or: pip install anthropic pytest
export ANTHROPIC_API_KEY=...  # only needed for live runs
for d in step*/; do (cd $d && python -m pytest -q); done   # each step is self-contained
```

Start with `step1_loop/LESSON.md`.
