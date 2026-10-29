# Step 1: the agent loop

## The one idea

An LLM can't do anything. It only reads text and writes text. An "agent" is a plain
program that lets the model *ask* for actions and then does them on its behalf:

```
messages = [user's task]
loop:
    response = model(system, tools, messages)
    append response as an assistant turn
    if the model asked for tools:
        run each one
        append the results as a user turn
    else:
        return the model's text
```

That's the whole of `Agent.send()` in `agent.py`. Claude Code, Codex and Claude Duet
all have this loop at their centre. Everything else (permissions, memory, subagents,
threads, routines) is engineering around it.

## What the model actually sees

Each request carries three things, and only these three:

| Part | In `agent.py` | What it does |
|---|---|---|
| `system` | `SYSTEM_PROMPT` | Who the model is and how it should work. Sent on every request. |
| `tools` | `TOOLS` | JSON schemas. The model never sees your Python, only the name, description and input schema. The description is the tool's manual. |
| `messages` | `self.messages` | The full conversation so far, every turn, every tool result. |

The API is stateless. The model remembers nothing between calls, so the harness resends
the whole history every time. That's why context fills up, and why step 3 (compaction,
caching) exists. Run with `--trace` and watch `messages` grow on each REQUEST.

## Why tool results go back as a user turn

The conversation must alternate: user, assistant, user, assistant. The model's
`tool_use` block is in an assistant turn, so the answer to it has to come in the next
user turn, as a `tool_result` block with the same `tool_use_id`. From the model's point
of view, "the user" is everything outside itself, including your harness.

Rules the API enforces:
- Every `tool_use` must get a `tool_result` in the very next user turn, or the next request fails.
- If the model asks for several tools at once, answer all of them in **one** user turn.
- A tool that fails still gets a result, with `is_error: true`. The model reads the error
  and tries something else. Never let a tool exception crash the loop.

## What `stop_reason` tells you

| Value | Meaning | What the loop does |
|---|---|---|
| `end_turn` | The model finished its answer. | Return the text to the user. |
| `tool_use` | The model wants one or more tools run. | Run them, append results, loop. |
| `max_tokens` | The answer hit `max_tokens` and was cut off. | If it was mid tool call, answer with an error so it can retry smaller; otherwise return what it said. |
| `refusal` | A safety classifier declined the request. | Stop and report it. |
| `pause_turn` | Only for server-side tools (web search etc.), which we don't use yet. | Not handled. |

## Two details worth noticing

1. **Append the response unchanged.** The assistant turn goes back exactly as it came,
   thinking blocks included. Opus 5.5 always thinks, and its thinking blocks are signed
   and tied to the conversation, so if you edit or drop earlier turns the model loses
   that reasoning. Harnesses are append-only for this reason.
2. **`--trace` shows reasoning.** The request sets `thinking.display = "summarized"`, so
   thinking blocks in the trace carry a readable summary instead of an empty string.

## Guardrails in this step (and their holes)

- File tools refuse paths outside `--cwd`.
- `run_bash` times out after 30 seconds, and long output is cut at 20,000 characters.
- `MAX_TURNS` stops a model that keeps calling tools forever.

The hole: `run_bash` can still `cd /` or `rm` anything the process can reach. There is
no permission prompt. Run it on a scratch folder only. Fixing this properly is step 2.

## Try it

```
cd harness/step1_loop
python -m pytest -q                 # 12 tests, no API key needed
python offline_demo.py              # a full traced run against a fake local API
python agent.py --cwd ./scratch --trace "create fizzbuzz.py and run it"   # live, needs a key
python agent.py --cwd ./scratch     # REPL
```

## Exercises

1. **Add a `grep` tool.** Add the schema to `TOOLS` and a method to `Workspace`. Then ask
   "find every TODO in this folder" with `--trace` and see whether the model picks `grep`
   or `run_bash`. Change the description and see if its choice changes.
2. **Count tokens and cost.** Each response has `usage.input_tokens` and
   `usage.output_tokens`. Keep a running total and print it after each `send()`, priced at
   $4 per million input and $20 per million output for Opus 5.5. Notice input tokens grow
   every turn even when you type little: that's the history being resent.
3. **Make it stream.** Swap `messages.create` for `client.messages.stream(...)`, print text
   as it arrives, and call `stream.get_final_message()` for the full response the loop
   needs. Nothing else in the loop should change.

Bonus: ask it to do something that fails (read a file that doesn't exist) and read in the
trace how the `is_error` result changes what it does next.

## Before step 2, make sure you can answer

Why does the harness, not the model, decide whether a tool actually runs? (Hint: look at
where `Workspace.run` is called. That one line is where permissions will go.)
