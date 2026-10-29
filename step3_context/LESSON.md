# Step 3: context management

## The one idea

The API is stateless. Every request carries the whole conversation, and the model knows
only what's in that one request. So three questions decide how well a long session goes:

| Question | Answer in this step | File |
|---|---|---|
| What should the model know before the conversation starts? | **Memory**: `HARNESS.md`, loaded once into the system prompt | `context.py` `load_memory` |
| How do we avoid paying full price to resend the same tokens? | **Caching**: mark the stable prefix; repeat requests read it from cache | `agent.py` `_call` |
| What happens when the conversation gets too long? | **Compaction**: summarize, then restart from the summary | `agent.py` `compact` |

Compare: `diff ../step2_permissions/agent.py agent.py`

## Memory

`HARNESS.md` in the project folder (and `~/.harness/HARNESS.md` for you as a user) is
read once at startup and appended to the system prompt. It works the same way as Claude
Code's `CLAUDE.md`. It's plain text the model reads every turn: how to run the tests,
conventions, pitfalls.

The model can add to it with the `remember` tool, which writes a line under `## Learned`.
Two choices are deliberate:

- **New notes take effect next session, not now.** Changing the system prompt mid-session
  would break the cache (below) and, on Opus 5.5, invalidate every earlier thinking block.
- **Memory is guarded like settings.** Whatever lands in `HARNESS.md` becomes part of every
  future system prompt, so a prompt-injected note would persist across sessions. `remember`
  always asks (even in accept-edits), and `write_file(HARNESS.md)` is a default deny rule.
  The system prompt also labels the notes as context, not as the user's instructions.

## Caching

Caching is a **prefix match**. The API renders `tools`, then `system`, then `messages`, and
caches up to each `cache_control` marker. If a later request starts with the same bytes,
those tokens are billed as cache reads: $0.20 per million instead of $4 on Opus 5.5, 20x
cheaper. Writing to the cache costs a little more than normal input ($5), once.

The harness uses two markers:
1. On the system prompt, which caches tools and system together.
2. A top-level `cache_control`, which follows the end of the conversation automatically, so
   each request reads everything the previous one wrote.

Then it follows one rule: **never change anything already sent.** A timestamp in the system
prompt, a re-sorted tool list or an edited old message breaks the prefix, and everything
after the change is billed at full price again. In an agent loop where each request resends
the whole history, this is the difference between paying for the conversation once and
paying for it on every turn. The meter after each reply shows it:

```
[5 requests | context now 1,013 tokens | input 4,069 (76% from cache) | output 176 | $0.0091 ($0.0198 without caching)]
```

## Compaction

When the context passes `--compact-at` tokens (default 200,000, though Opus 5.5 holds 1M:
long contexts cost more per request, and the model's focus on any one part thins out), the
harness:

1. Sends the conversation **as is**, plus a summary instruction appended to the newest user
   turn. Because the prefix is unchanged, almost all of it is a cache read, so the summary
   is cheap.
2. Replaces the whole history with one user turn: the summary, plus whatever was waiting
   (your new message, or the last tool result if it happened mid-task).

This is "simple compaction", and the shape matters more than it looks. Opus 5.5's thinking
blocks are signed against the exact history that produced them. The tempting alternative,
summarizing old turns but keeping the last few verbatim ("keep-tail"), sends thinking blocks
that no longer sit next to their original history. On newer accounts that's a 400 error. So:
summarize everything, replay nothing, and never compact while a `tool_use` is waiting for its
result. The summary prompt in `context.py` is Anthropic's recommended one. Its last sentence
("do not call any tools") matters because the summary request still carries the tools.

The API also offers server-side compaction and context editing (clearing old tool results).
They're better in production. We do it by hand here so you can see every piece.

## Try it

```
cd harness/step3_context
python -m pytest -q                    # 48 tests
python offline_demo.py                 # memory, cache hits and a compaction, with a simulated cache
python agent.py --cwd ./scratch --compact-at 20000    # live; /cost and /compact in the REPL
```

In the demo, look at request 6 (the summary): nearly all of it is a cache read. Then look at
request 7: only the tools and system prompt are still cached, because compaction replaced
the conversation. Every compaction restarts the cache, so it should be rare.

## Exercises

1. **Find a cache breaker.** Add the current time to the system prompt and run the demo
   again. Watch "from cache" fall. Then move the time into the user's message instead and
   see it recover. This is the most common caching bug in real harnesses.
2. **Clear old tool results.** Before compacting, try a cheaper fix: long `read_file` and
   `run_bash` outputs from many turns ago are rarely needed. Look up context editing
   (`clear_tool_uses_20250919`) in the API docs and add it as a `--clear-old-results` flag.
   Why is doing this server-side safe for thinking blocks, when editing the history yourself isn't?
3. **Close the hole.** `write_file(HARNESS.md)` is denied, but `run_bash("echo x >> HARNESS.md")`
   works, because the sandbox lets bash write anywhere in the project. Fix it, either in the
   permission rules or in the sandbox. Which is the more reliable place?

## Before step 4, make sure you can answer

Why does a single changed byte near the start of a request cost more than one changed byte
near the end?
