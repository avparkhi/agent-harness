"""The whole eval workflow, with no API key: validate the cases, run, compare, and see
why one run of each version isn't enough to tell them apart.

The "model" here is fakes.flaky: it fixes easy bugs 90% of the time, medium 60%, hard 30%.
Both runs below use the *same* fake, so any difference between them is pure noise.

Run:  python offline_demo.py
"""
from types import SimpleNamespace

import evals
import fakes


def config(**overrides):
    base = dict(agent=str(evals.DEFAULT_AGENT), model="claude-opus-5-5", effort=None, max_turns=20,
                no_subagents=False, system_extra="", no_sandbox=False, parallel=4, trials=1)
    return SimpleNamespace(**{**base, **overrides})


def quiet_run(cases, cfg, make_client):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return evals.run(cases, cfg, make_client)


if __name__ == "__main__":
    cases = evals.load_cases()
    print("1. Are the cases sound? (hidden tests must fail on the bug and pass on the fix)")
    evals.validate(cases)

    print("\n2. Same agent, run twice, one trial each:")
    a = quiet_run(cases, config(), fakes.factory("flaky", cases, seed=1))
    b = quiet_run(cases, config(), fakes.factory("flaky", cases, seed=2))
    print(evals.compare(a, b, "run 1", "run 2"))
    print("   Nothing changed between these runs, yet cases flipped. One run tells you little.")

    print("\n3. Same agent, 5 trials each:")
    a = quiet_run(cases, config(trials=5), fakes.factory("flaky", cases, seed=1))
    b = quiet_run(cases, config(trials=5), fakes.factory("flaky", cases, seed=2))
    print(evals.compare(a, b, "run 1", "run 2"))

    print("\n4. A real difference: the flaky agent against one that gives up:")
    c = quiet_run(cases, config(trials=5), fakes.factory("noop", cases))
    print(evals.compare(a, c, "flaky", "gives up"))
