"""Step 5: measure the agent instead of guessing.

Every change you make to a harness (a prompt line, a tool description, effort, subagents
on or off) can help one task and hurt another. Trying it on one example and eyeballing the
result tells you almost nothing. An eval is a fixed set of tasks with automatic grading, run
the same way every time, so two versions of the agent can be compared on numbers.

Each case in cases/<name>/ has:
    case.json   the bug report the agent sees, and a difficulty label
    repo/       the buggy code the agent starts with
    hidden/     tests the agent never sees, run after it finishes: the grader
    solution/   a reference fix, used only to check the case itself is sound

Commands:
    python evals.py validate                               # are the cases sound? no API key needed
    python evals.py run --name baseline --trials 3         # run the agent on every case
    python evals.py run --name low-effort --effort low --trials 3
    python evals.py compare baseline low-effort            # what changed, and is it noise?
    python evals.py run --name demo --fake flaky --trials 3   # whole pipeline, no API key
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES_DIR = HERE / "cases"
RESULTS_DIR = HERE / "results"
DEFAULT_AGENT = HERE.parent / "step4_subagents"
GRADE_TIMEOUT = 60

INSTRUCTIONS = ("\n\nFix the bug in this repository. Keep function names and signatures as they are. "
                "When you're done, reply with one sentence describing the fix.")


@dataclass
class Case:
    name: str
    difficulty: str
    task: str
    path: Path


def load_cases(pattern: str = "*") -> list[Case]:
    cases = []
    for d in sorted(CASES_DIR.glob(pattern)):
        meta = json.loads((d / "case.json").read_text())
        cases.append(Case(d.name, meta["difficulty"], meta["task"], d))
    return cases


# --- grading --------------------------------------------------------------------------------

def run_hidden_tests(workdir: Path, case: Case, sandbox=None, env=None) -> tuple[bool, str]:
    """Copy the hidden tests in *after* the agent has finished, run them, report pass/fail.
    The agent never saw these tests, so it can't have tuned its fix to them."""
    hidden = workdir / ".eval_hidden"
    if hidden.exists():
        shutil.rmtree(hidden)     # the agent can't plant its own "hidden" tests either
    shutil.copytree(case.path / "hidden", hidden)
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(hidden)]
    if sandbox is not None:       # the code under test was written by the agent: sandbox it too
        argv = sandbox.wrap(argv, workdir)
    try:
        r = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, timeout=GRADE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return False, f"grader timed out after {GRADE_TIMEOUT}s"
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def fresh_repo(case: Case, apply_solution: bool = False) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix=f"eval-{case.name}-"))
    shutil.copytree(case.path / "repo", workdir, dirs_exist_ok=True)
    if apply_solution:
        shutil.copytree(case.path / "solution", workdir, dirs_exist_ok=True)
    return workdir


def validate(cases: list[Case]) -> bool:
    """A case is only useful if its grader can tell a fix from no fix: the hidden tests must
    fail on the original bug and pass on the reference solution. Check this before trusting
    any score."""
    ok = True
    for case in cases:
        bug_dir, fix_dir = fresh_repo(case), fresh_repo(case, apply_solution=True)
        fails_on_bug = not run_hidden_tests(bug_dir, case)[0]
        passes_on_fix, output = run_hidden_tests(fix_dir, case)
        sound = fails_on_bug and passes_on_fix
        ok &= sound
        print(f"  {'ok ' if sound else 'BAD'} {case.name:18} fails on bug: {fails_on_bug!s:5}  passes on fix: {passes_on_fix}")
        if not passes_on_fix:
            print("      " + output.strip().replace("\n", "\n      "))
        shutil.rmtree(bug_dir), shutil.rmtree(fix_dir)
    return ok


# --- running the agent ----------------------------------------------------------------------

def load_agent(agent_dir: Path):
    """Import agent.py, permissions.py and sandbox.py from one step's folder."""
    sys.path.insert(0, str(agent_dir.resolve()))
    optional = lambda name: importlib.import_module(name) if (agent_dir / f"{name}.py").exists() else None
    return importlib.import_module("agent"), optional("permissions"), optional("sandbox")


def run_attempt(case: Case, trial: int, client, mods, cfg) -> dict:
    agent_mod, permissions_mod, sandbox_mod = mods
    workdir = fresh_repo(case)
    sandbox = sandbox_mod.pick_sandbox(not cfg.no_sandbox) if sandbox_mod else None
    started = time.time()
    record = {"case": case.name, "difficulty": case.difficulty, "trial": trial, "error": None}
    try:
        ws = agent_mod.Workspace(workdir, sandbox) if "sandbox" in inspect.signature(agent_mod.Workspace).parameters \
            else agent_mod.Workspace(workdir)
        # Headless: nobody is there to approve, so bypass mode. The sandbox is the safety boundary.
        options = {"permissions": permissions_mod.Permissions.load(workdir, mode="bypass") if permissions_mod else None,
                   "model": cfg.model, "max_turns": cfg.max_turns, "log": lambda *_: None}
        tools = [t for t in getattr(agent_mod, "TOOLS") if not (cfg.no_subagents and t["name"] == "task")]
        options["tools"] = tools
        if cfg.system_extra:
            options["system_prompt"] = agent_mod.SYSTEM_PROMPT + "\n" + cfg.system_extra.replace("{", "{{").replace("}", "}}")
        accepted = inspect.signature(agent_mod.Agent).parameters      # older steps take fewer options
        agent = agent_mod.Agent(client, ws, **{k: v for k, v in options.items() if k in accepted})
        final = agent.send(case.task + INSTRUCTIONS)
        record["final"] = final[:300]
        meter = getattr(agent, "meter", None)
        if meter is not None:
            record.update(requests=meter.requests, cost=round(meter.cost, 5),
                          input_tokens=meter.input + meter.cache_write + meter.cache_read, output_tokens=meter.output)
    except Exception as e:      # an API error or a harness crash is a failed attempt, not a crashed eval
        record["error"] = f"{type(e).__name__}: {e}"[:500]
    record["agent_seconds"] = round(time.time() - started, 1)
    record["passed"], record["grader_output"] = (False, "not graded") if record["error"] else \
        run_hidden_tests(workdir, case, sandbox, sandbox_mod.clean_env() if sandbox_mod else None)
    shutil.rmtree(workdir, ignore_errors=True)
    return record


def run(cases: list[Case], cfg, make_client) -> list[dict]:
    mods = load_agent(Path(cfg.agent))
    if cfg.effort:
        mods[0].EFFORT = cfg.effort
    attempts = [(case, trial) for case in cases for trial in range(1, cfg.trials + 1)]
    results = []
    with ThreadPoolExecutor(max_workers=cfg.parallel) as pool:
        futures = [pool.submit(run_attempt, case, trial, make_client(case, trial), mods, cfg) for case, trial in attempts]
        for future in futures:
            r = future.result()
            results.append(r)
            mark = "pass" if r["passed"] else ("ERROR" if r["error"] else "fail")
            print(f"  {r['case']:18} trial {r['trial']}  {mark:5}  ${r.get('cost', 0):.4f}  "
                  f"{r.get('requests', '?')} requests  {r['agent_seconds']}s")
    return results


# --- reporting ------------------------------------------------------------------------------

def by_case(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["case"], []).append(r)
    return grouped


def summarize(results: list[dict]) -> dict:
    n = len(results)
    grouped = by_case(results)
    return {
        "attempts": n,
        "pass_rate": sum(r["passed"] for r in results) / n if n else 0.0,
        # Did it solve the case in *any* trial, and in *every* trial? The gap is flakiness.
        "solved_any": sum(any(r["passed"] for r in rs) for rs in grouped.values()),
        "solved_all": sum(all(r["passed"] for r in rs) for rs in grouped.values()),
        "cases": len(grouped),
        "errors": sum(bool(r["error"]) for r in results),
        "cost_per_attempt": sum(r.get("cost", 0) for r in results) / n if n else 0.0,
        "total_cost": sum(r.get("cost", 0) for r in results),
    }


def report(results: list[dict]) -> str:
    s = summarize(results)
    lines = [f"{'case':18} {'difficulty':10} passed"]
    for name, rs in by_case(results).items():
        lines.append(f"{name:18} {rs[0]['difficulty']:10} {sum(r['passed'] for r in rs)}/{len(rs)}")
    lines.append(f"\npass rate {s['pass_rate']:.0%} over {s['attempts']} attempts | solved in every trial: "
                 f"{s['solved_all']}/{s['cases']} | in at least one: {s['solved_any']}/{s['cases']} | "
                 f"errors {s['errors']} | ${s['cost_per_attempt']:.4f} per attempt, ${s['total_cost']:.4f} total")
    return "\n".join(lines)


def compare(a: list[dict], b: list[dict], name_a: str, name_b: str) -> str:
    """Side by side, plus a rough check of whether the difference is bigger than noise."""
    ga, gb = by_case(a), by_case(b)
    lines = [f"{'case':18} {name_a:>12} {name_b:>12}"]
    for name in sorted(set(ga) | set(gb)):
        ra, rb = ga.get(name, []), gb.get(name, [])
        cell = lambda rs: f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "-"
        flag = "  <- changed" if ra and rb and (sum(r["passed"] for r in ra) / len(ra)) != (sum(r["passed"] for r in rb) / len(rb)) else ""
        lines.append(f"{name:18} {cell(ra):>12} {cell(rb):>12}{flag}")
    sa, sb = summarize(a), summarize(b)
    delta = sb["pass_rate"] - sa["pass_rate"]
    # Standard error of a difference between two pass rates. With a few dozen attempts it's
    # large, which is the honest answer: small evals can only detect big changes.
    se = math.sqrt(sa["pass_rate"] * (1 - sa["pass_rate"]) / max(sa["attempts"], 1)
                   + sb["pass_rate"] * (1 - sb["pass_rate"]) / max(sb["attempts"], 1))
    verdict = ("identical" if delta == 0 else "likely a real difference" if abs(delta) > 2 * se else
               "within noise: run more trials or add cases before trusting it")
    lines.append(f"\npass rate {sa['pass_rate']:.0%} -> {sb['pass_rate']:.0%} ({delta:+.0%}, +/-{2 * se:.0%}): {verdict}")
    lines.append(f"cost per attempt ${sa['cost_per_attempt']:.4f} -> ${sb['cost_per_attempt']:.4f}")
    return "\n".join(lines)


def save(results: list[dict], name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in results))
    return path


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS_DIR / f"{name}.jsonl").read_text().splitlines() if line]


# --- command line ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the agent on a set of bugs with hidden tests.")
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="check each case: hidden tests fail on the bug and pass on the fix")
    v.add_argument("--cases", default="*", help="glob over case names")
    r = sub.add_parser("run", help="run the agent on the cases")
    r.add_argument("--name", required=True, help="results are saved as results/NAME.jsonl")
    r.add_argument("--cases", default="*")
    r.add_argument("--trials", type=int, default=1, help="attempts per case; models vary run to run")
    r.add_argument("--agent", default=str(DEFAULT_AGENT), help="folder of the step to evaluate")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--max-turns", type=int, default=20)
    r.add_argument("--no-subagents", action="store_true")
    r.add_argument("--system-extra", type=Path, help="file with text to append to the system prompt")
    r.add_argument("--no-sandbox", action="store_true")
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--fake", choices=["oracle", "flaky", "noop"], help="scripted model, no API key")
    r.add_argument("--seed", type=int, default=0, help="random seed for --fake flaky")
    r.add_argument("--yes", action="store_true", help="don't ask before spending money")
    c = sub.add_parser("compare", help="compare two saved runs")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)

    if args.command == "validate":
        cases = load_cases(args.cases)
        print(f"Checking {len(cases)} cases:")
        sys.exit(0 if validate(cases) else 1)
    if args.command == "compare":
        print(compare(load(args.a), load(args.b), args.a, args.b))
        return

    cases = load_cases(args.cases)
    args.system_extra = args.system_extra.read_text() if args.system_extra else ""
    if args.fake:
        import fakes
        make_client = fakes.factory(args.fake, cases, args.seed)
    else:
        attempts = len(cases) * args.trials
        print(f"This runs {attempts} agent attempts against the real API ({args.model}) and costs real money.")
        if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
            return
        import anthropic
        shared = anthropic.Anthropic()
        make_client = lambda case, trial: shared
    print(f"Running {len(cases)} cases x {args.trials} trials ({args.name}):")
    results = run(cases, args, make_client)
    path = save(results, args.name)
    print("\n" + report(results))
    print(f"\nsaved {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
