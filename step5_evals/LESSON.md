# Step 5: evals

## The one idea

Steps 1 to 4 added features. Each one *seemed* to help. Step 5 is how you find out.
An eval is a fixed set of tasks with automatic grading, run the same way every time, so
two versions of your agent can be compared on numbers instead of impressions.

This is the step that separates harness engineering from prompt fiddling. Every serious
agent team works this way: change one thing, run the eval, keep the change only if the
numbers say so.

## What's here

| File | What it does |
|---|---|
| `cases/<name>/` | 8 small, realistic bugs: off-by-one, mutable default, CSV quoting, path traversal, swallowed retries... |
| `evals.py` | `validate`, `run` and `compare` commands |
| `fakes.py` | scripted models (oracle, flaky, noop) so you can run everything without an API key |
| `offline_demo.py` | the whole workflow in one go, including why one run isn't enough |

Each case has four parts:

```
case.json   the bug report the agent sees (written like a real issue, not a hint)
repo/       the buggy code it starts with
hidden/     tests it never sees, copied in after it finishes: the grader
solution/   a reference fix, only used to check the case itself
```

## Four rules the runner follows

**1. Grade on hidden tests, never on what the agent says.** The flaky fake always replies
"Fixed the bug." Sometimes it changed nothing. The test `test_claiming_a_fix_is_not_a_fix`
shows that only the hidden tests decide. The runner also deletes any `.eval_hidden` folder
the agent made before copying the real tests in, so it can't grade itself.

**2. Validate the eval before trusting it.** `python evals.py validate` checks that each
case's hidden tests *fail* on the bug and *pass* on the reference fix. A case that fails
this measures nothing. This caught a real bug while I was building it: the first
`dedupe_order` test used email strings, and `list(set(...))` sometimes kept their order by
luck, because Python randomizes string hashing per process. On that run the hidden test
passed on the buggy code, so the case would have scored a non-fix as a fix. The fix was a
test with small integers, whose set order is fixed.
Flaky graders are the most common way evals lie.

**3. Run more than once.** Models vary from run to run. `--trials 3` runs each case three
times and reports two numbers: solved in every trial (reliable), and solved in at least one
(possible). The gap between them is flakiness.

**4. Know your noise.** `compare` doesn't just show the difference. It estimates whether
it's bigger than chance (two standard errors). Run the demo: the *same* flaky agent, run
twice, flips 4 of 8 cases and drops 25 points, and `compare` correctly calls it noise. With
8 cases you can only detect big effects. To see a 5-point improvement reliably you need
hundreds of attempts. That's the honest maths, and it's why real eval sets keep growing.

## How the runner treats the agent

- Each attempt gets a fresh copy of the repo in a temp folder.
- Permissions are set to bypass, because nobody is there to approve. The sandbox from
  step 2 is the safety boundary, and the hidden tests run inside it too, since the code
  under test was written by the model.
- An API error or a harness crash counts as a failed attempt with the error recorded. The
  eval itself never crashes halfway through.
- Attempts run in parallel (`--parallel 4`). Results go to `results/NAME.jsonl`, one line
  per attempt, so you can dig into any failure later.
- `--agent ../step3_context` evaluates an earlier step, so you can measure what each layer
  is actually worth.

## Try it

```
cd harness/step5_evals
python -m pytest -q                       # 11 tests
python evals.py validate                  # no API key needed
python offline_demo.py                    # validate, noise, and a real difference

# With an API key (it asks before spending money):
python evals.py run --name baseline --trials 3
python evals.py run --name effort-low --effort low --trials 3
python evals.py compare baseline effort-low
```

I couldn't run it against the real API here, so I don't have real scores for you yet. Your
first `baseline` run will be the first real number for your harness. 8 cases x 3 trials is
24 attempts. The runner prints the cost per attempt.

## Experiments worth running

1. **Effort.** `--effort low` vs `high` vs `xhigh`. Does `low` hold up on the easy cases?
   Is the hard case (`retry`) where the extra thinking pays off? Compare cost per *passed*
   case, not per attempt.
2. **Each layer's value.** Run `--agent ../step1_loop` and `--agent ../step4_subagents`. Does
   the extra machinery help on bugs this small, or only cost more? (It may not help, and
   that's a useful finding: it tells you which kind of task each layer is for.)
3. **One prompt line.** Put "Before changing code, write a failing test that reproduces the
   bug." in a file and pass it with `--system-extra`. Measure it instead of guessing.

## Exercises

1. **Add three cases from your own work.** Take bugs you've actually fixed, strip them down,
   and write hidden tests. Run `validate` until all are sound. Your own bugs are a better
   eval than anyone else's.
2. **Grade more than pass/fail.** Add a second check: did the agent change files it didn't
   need to (diff the repo against the solution's file list)? A "fix" that rewrites half the
   repo should score worse.
3. **Track trajectories.** Save the full message history of each attempt (the trace from
   step 1) next to the result, then find the failing runs and read what the model was
   thinking. Most harness improvements come from reading failures, not from scores.

## Before step 6, make sure you can answer

Your new prompt line takes the pass rate from 70% to 78% on one run of 8 cases. What do you
do before you keep it?
