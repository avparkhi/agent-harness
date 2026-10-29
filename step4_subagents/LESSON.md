# Step 4: planning and subagents

## The one idea

A long task fails in two ways: the agent loses track of the plan, or its context fills up
with search results and file contents it no longer needs. This step adds one tool for each.

| Tool | File | Fixes |
|---|---|---|
| `todo_write` | `planning.py` | Losing the plan. The model writes its plan down and re-reads it every turn. |
| `task` | `subagents.py` | Context filling up. A subagent does the messy work in its own context and returns a short report. |

Plus `search_files`, a read-only search tool, so read-only subagents can explore without bash.
Compare: `diff ../step3_context/agent.py agent.py`

## Planning: the harness doesn't plan, the model does

`todo_write` takes the *whole* list every time: `[{content, status}]`, with status
`pending`, `in_progress` or `completed`. The harness checks it (only one `in_progress`),
shows it to you, and echoes it back in the tool result.

That's the whole trick. The plan now lives in the conversation, so on every turn the model
re-reads what it decided and where it is. Nothing in the harness executes the plan. It's a
scratchpad the model is prompted to use. Two harness-side touches make it sturdier:

- The list is also kept as harness state (`agent.todos`), so when compaction replaces the
  conversation, the exact list is put back into the continuation instead of trusting the summary.
- The tool description and system prompt say when to use it (3+ steps), and to mark items
  completed one at a time. Behaviour like this comes from tool descriptions as much as code.

## Subagents: same loop, fresh context

A subagent is just another `Agent`, created inside a tool call:

```
main agent calls task(description, prompt, agent_type)
    -> run_subagent() builds a new Agent:
         its own empty conversation, starting with the prompt (the "brief")
         its own system prompt and a smaller tool list
         permissions derived from the parent's
    -> the child runs the same loop until it answers
    -> its final text becomes the tool_result the parent sees
```

Run the demo and look at the request sizes: each subagent read files and search results,
but the main agent's context only grew by the two short reports. That's **context
isolation**, and it's why subagents exist. Big searches happen somewhere else, and the main
conversation stays small, cheap and focused.

The two types:

| Type | Tools | Permissions | Use for |
|---|---|---|---|
| `explore` | read_file, search_files | parent's rules, forced to read-only mode | finding and reading code |
| `general` | read, write, bash, search | exactly the parent's (approvals still come to you) | independent pieces of work |

Neither gets `task`, so subagents can't start subagents. Without that limit, one bad brief
can fan out into an expensive tree.

## Parallelism

When the model calls `task` several times in one turn, `_run_tools` runs those calls in a
thread pool and returns all results, in the order asked, in one user turn. Approval prompts
take turns behind a lock. The test `test_parallel_subagents_run_at_the_same_time` proves
the calls overlap: each fake subagent waits at a barrier that only opens when both are running.

## The costs (what the brief and the report have to carry)

- **The brief is everything.** The subagent can't see the conversation. A vague brief like
  "check the config" gets vague work. That's why the tool description demands a complete,
  self-contained prompt. It's the same skill as briefing a colleague.
- **The report is lossy.** The parent knows only what the child chose to write down, so
  the prompt asks for paths, line numbers and uncertainties, and the system prompt tells
  the parent to check reports before relying on them.
- **It costs more in total.** Each subagent pays for its own system prompt and its own reads.
  The meter now shows how much went to subagents. Delegate when isolation or parallelism is
  worth it, not by default.

`--subagent-model` lets subagents use a different model. Claude Code uses a smaller, faster
model for some of its built-in agents. Try `claude-haiku-5-5` for explore and compare
cost and quality.

## Try it

```
cd harness/step4_subagents
python -m pytest -q                    # 60 tests
python offline_demo.py                 # a plan, two parallel explore subagents, a summary
python agent.py --cwd ./some/real/repo --mode read-only "explain how this project is structured"
```

## Exercises

1. **Background subagents.** Right now the parent waits for every subagent to finish. Make
   `task` return immediately with an id, deliver the report as a user message when it's
   ready, and add a `wait_for` tool. Anthropic's guidance says agents finish sooner this way,
   because the parent keeps working. Which part of the loop has to change?
2. **Nudge on unfinished plans.** If the model ends its turn while todos are still pending,
   the harness could append a reminder and continue. Add it with a limit of one nudge per
   task. Why must the reminder be *appended* rather than inserted earlier (think step 3)?
3. **Custom agent types.** Load extra subagent types from `.harness/agents/*.md` (name,
   tools and system prompt in the file), like Claude Code's `.claude/agents/`. Which
   permission rule from step 2 must cover that folder, and why?

## Before step 5, make sure you can answer

The main agent and its subagent run the same loop code. What exactly is different between
them, and which of those differences is what keeps the main context small?
