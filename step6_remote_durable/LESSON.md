# Step 6: durable sessions and remote sandboxes

## The one idea

Up to step 5, a session lived in one Python process on your machine. Close the laptop,
lose the network, or hit Ctrl-C at the wrong moment, and the work is gone. Worse, a tool
call that was halfway through leaves no trace, so you can't even tell what happened.

This step separates three things that were tangled together:

| Thing | Where it lives now | Survives |
|---|---|---|
| The conversation | an append-only log in SQLite (`durable.py`) | the process dying |
| The files and shell | a Daytona sandbox (`remote.py`) | the process dying, and your machine being off |
| The loop and your API key | your machine, as before | nothing, and that's fine: it rebuilds from the other two |

Once the state is in the log and the work is in the sandbox, the process that runs the
loop is disposable. That is the property behind Claude Code's `--resume`, cloud agents
that keep working after you close the tab, and agents that run for hours.

Compare: `diff ../step4_subagents/agent.py agent.py`

## Durable sessions: six rules

The design follows pi's durable package
([earendil-works/pi, packages/durable](https://github.com/earendil-works/pi/tree/main/packages/durable)),
cut down to its core. Each rule is a few lines in `durable.py` or `Agent`:

| Rule | Where | What breaks without it |
|---|---|---|
| **1. Commit before you show.** | `Agent._commit`, called before every print, return and tool run | The screen says "done" but the log doesn't, so a resume redoes it. |
| **2. Commits are atomic.** | `Store.tx` | A crash between "assistant answered" and "done" leaves a half state. The final answer and the `final` entry land together, and a tool result lands with the plan it set. |
| **3. Intent before effect.** | `_run_tool` commits `intent` before running | After a crash you can't tell "never started" from "was running". |
| **4. Only replay what's safe.** | `REPLAY_SAFE`, `Agent.resume` | Rerun `git push` or `npm publish` blindly and you do it twice. |
| **5. Don't crash-loop.** | `Session.count_restart`, `RESTART_LIMIT` | A task that crashes the harness every time (a huge file, a bad command) restarts forever. After 5 tries it's parked until you `--force` it. |
| **6. One writer.** | the lease: `acquire`, `heartbeat`, and the check inside `append` | Two processes resume the same session and interleave their commits. |

### What a resume does

`fold()` rebuilds the conversation from the log, then looks at the last tool round:

```
call finished before the crash      -> keep its stored result, don't rerun
call started, tool is replay-safe   -> run it again (reading a file twice is harmless)
call started, tool is not           -> answer "Interrupted: may or may not have taken effect"
call never started                  -> run it normally (it still goes through permissions)
```

The interrupted answer is the important one. The harness doesn't know whether
`make deploy` finished, so it doesn't guess. It tells the model, which can check
(`git log`, `ls dist/`) before deciding. This is the same idea as step 2: a failure goes
back as a tool result, and the model adapts.

On a remote sandbox this matters even more. If your machine dies mid-command, the command
**keeps running on the sandbox**. The demo shows this: the killed child's `sleep 5`
outlives it. "May or may not have taken effect" is the honest description.

### Things that must not change on resume

- **The system prompt.** It's stored in the session and reused exactly. Rebuilding it
  (with a HARNESS.md that changed since) would miss the cache and break the thinking
  history. `test_the_system_prompt_is_frozen_with_the_session` checks this.
- **Thinking blocks.** The assistant turns are stored exactly as they came back,
  signatures included, so the request after a restart is byte-identical to the one that
  would have been sent without the crash. The demo prints this check.

### Exactly-once submits

`send(prompt, request_id=...)`: if the id was already answered, you get the stored
answer back and no model call is made. If it was submitted but not finished, it resumes.
Use it when something outside retries for you (a webhook, a queue, a flaky network).

## Remote sandboxes: one interface, two backends

`DaytonaWorkspace` has the same methods as `Workspace` (`read_file`, `write_file`,
`run_bash`, `search_files`, `memory`, `remember`), so the Agent can't tell where its
tools run. What crosses to the sandbox is only tool calls, after the permission check:

```
your machine                          Daytona sandbox
the loop, permissions, approvals  ->  read_file / write_file / run_bash / search_files
ANTHROPIC_API_KEY (never sent)        the working directory, HARNESS.md
the session log                       processes that outlive your laptop
```

Details worth reading in `remote.py`:

- Paths are checked as strings (`posixpath`), not with `Path.resolve()`, which would look
  at your local disk.
- `search_files` runs a small script on the sandbox, with the pattern passed in an
  environment variable, never pasted into code. `test_a_hostile_search_pattern_is_just_a_pattern`
  tries to break out.
- The session stores the sandbox id. `--resume` calls `daytona.get(id)` and starts the
  sandbox if it auto-stopped (`AUTO_STOP_MINUTES = 30`). An idle sandbox stops costing
  compute, and its disk is kept.
- Your local `.harness/settings.json` still sets the permission rules. The sandbox is a
  second boundary, not a replacement for the first.

## Try it

```
cd harness/step6_remote_durable
python -m pytest -q                 # 84 tests, no keys needed
python offline_demo.py              # kill -9 mid-command, then resume in a new process
```

Live, with your keys:

```
python -m pip install daytona
export ANTHROPIC_API_KEY=...  DAYTONA_API_KEY=...

# durable, local tools
python agent.py --cwd ./myproject --durable "add type hints to utils.py"

# durable, tools in a Daytona sandbox, your project copied in, results copied back
python agent.py --cwd ./myproject --remote --durable --upload --download ./myproject-out "run the tests and fix what fails"

python agent.py --cwd ./myproject --list            # sessions and where each one stands
python agent.py --cwd ./myproject --show ID         # the log, one line per entry
python agent.py --cwd ./myproject --resume ID       # carry on (reconnects to the sandbox)
```

Kill it with Ctrl-C or `kill -9` partway through, then `--resume`. Add
`.harness/sessions.db` to your project's `.gitignore`. A non-durable `--remote` run deletes
its sandbox at the end, and a durable one keeps it so you can resume. Delete old sandboxes
from the Daytona dashboard.

I couldn't run this against a real Daytona account or the real API here. The remote code
is tested against `fake_daytona.py`, which mimics the SDK calls `remote.py` uses
(`create`, `get`, `start`, `fs.upload_file`, `fs.download_file`, `process.exec`). The
parameters are built with the real SDK's `CreateSandboxFromSnapshotParams` (version
0.230). Your first live `--remote` run is the real test.

## What pi does that this doesn't

pi models every step as a durable *task* with phases that rerun from the top after a
restart, plus typed documents, hooks, forks, steering a busy conversation, and streamed
output persisted within a second. This step keeps the parts that make the idea click.
pi's README and spec are worth reading next to `durable.py`, especially the parts on tasks
and phases.

## Exercises

1. **Durable subagents.** A subagent interrupted mid-run comes back as "interrupted",
   because `task` isn't replay-safe. Give each child its own session, linked to the
   parent's call id, so a resume picks the child up where it stopped. What should happen
   to a child whose parent was aborted?
2. **Run the whole loop in the sandbox.** Right now the loop stays on your machine. Move it
   into the sandbox (Daytona secrets can hold the API key) so you can close your laptop
   for good. What does that cost in security, and where does the session log live now?
   Why does the lease suddenly matter much more?
3. **Per-call replay hints.** `run_bash("pytest -q")` is safe to rerun and
   `run_bash("git push")` is not, but both count as one tool. Let the model mark a call as
   safe to repeat (an extra `idempotent` field), and decide whether you trust that.
4. **Streaming without losing it.** Switch to streaming responses, show text as it
   arrives, and commit the partial answer at most every second, like pi's
   `persistWithinMs`. What should a resume do with a half-finished answer?

## Before step 7, make sure you can answer

Your agent ran `npm publish` on a remote sandbox, and your laptop died before the result
came back. On resume, why does the harness tell the model "interrupted" instead of either
rerunning the command or treating it as failed, and what should the model do next?
